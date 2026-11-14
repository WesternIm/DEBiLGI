"""Human propositions and scoped implications, independent of reply wording.

Only linguistic operators are built in. Predicates, states and consequences
come from observed messages. Forward chaining requires every premise; it never
uses missing evidence as negation or treats a capability as an actual action.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, replace
from functools import lru_cache

from debilgi.language.dialogue_obligations import thought_from_data
from debilgi.language.morphology import DEFAULT_MORPHOLOGY
from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    SemanticDependency,
)
from debilgi.language.structured_utterance import TokenInterpretation, analyze_structured_utterance
from debilgi.language.tokenizer import tokenize

_PERSONS = {"я": "ты", "ты": "я", "мы": "вы", "вы": "мы"}
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


@dataclass(frozen=True, slots=True)
class MeaningProof:
    thought: ClauseThought
    source_event_ids: tuple[str, ...]
    rule_ids: tuple[str, ...] = ()
    premise_keys: tuple[tuple, ...] = ()


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


def _lexeme(token: TokenInterpretation, perspective: str) -> ConceptLexeme:
    lemma = token.lemma
    surface = token.corrected_form or token.surface
    features = dict(token.grammatical_features)
    if perspective == "user" and lemma in _PERSONS:
        lemma = surface = _PERSONS[lemma]
        features["person"] = "first" if lemma in {"я", "мы"} else "second"
    return ConceptLexeme(
        lemma, surface, token.part_of_speech, tuple(features.items()), token.entity_kind
    )


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
    text: str,
    *,
    perspective: str = "user",
    fallback_subject: ConceptArgument | None = None,
    allow_question: bool = False,
) -> ClauseThought | None:
    """Parse a complete simple proposition, including zero copula and modality."""
    structure = analyze_structured_utterance(text.strip(" ,.;:!?"))
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
    if any(t.lemma in _UNSUPPORTED or t.surface in {"«", "»", '"'} for t in tokens):
        return None
    finite = [t for t in tokens if t.part_of_speech == "verb" and "VERB" in t.grammemes]
    if len(finite) > 1:
        return None
    predicate = finite[0] if finite else None
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
            _lexeme(subject_token, perspective), grammatical_case="nominative"
        )
        used.add(subject_token.index)
        if (
            predicate
            and dict(predicate.grammatical_features).get("person") in {"first", "second"}
            and subject_token.part_of_speech != "pronoun"
        ):
            return None
    elif fallback_subject:
        subject = fallback_subject
    elif predicate and dict(predicate.grammatical_features).get("person") in {"first", "second"}:
        features = dict(predicate.grammatical_features)
        personal = {
            ("first", "singular"): "я",
            ("first", "plural"): "мы",
            ("second", "singular"): "ты",
            ("second", "plural"): "вы",
        }.get((features.get("person"), features.get("number")))
        if not personal:
            return None
        personal = _PERSONS.get(personal, personal) if perspective == "user" else personal
        subject = ConceptArgument(ConceptLexeme(personal, personal, "pronoun"))
    else:
        return None
    dependencies = [SemanticDependency("subject", subject)]
    if predicate:
        used.add(predicate.index)
        predicate_lexeme = _lexeme(predicate, perspective)
        predicate_class = "modal" if predicate.lemma in _MODALS else "lexical"
    else:
        predicate_lexeme = ConceptLexeme("быть", "быть", "verb", (("tense", "present"),))
        predicate_class = "copular"
    infinitives = [t for t in tokens if "INFN" in t.grammemes and t.index not in used]
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
            SemanticDependency("action_complement", ConceptArgument(_lexeme(action, perspective)))
        )
        used.add(action.index)
    elif infinitives:
        return None

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
                    _lexeme(head, perspective),
                    tuple(_lexeme(t, perspective) for t in modifiers),
                    preposition=preposition,
                    grammatical_case=dict(head.grammatical_features).get("case"),
                ),
            )
        )
        used.update((head.index, *(t.index for t in modifiers)))
        return True

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
                    _lexeme(prop, perspective),
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
                SemanticDependency("manner", ConceptArgument(_lexeme(token, perspective)))
            )
        elif token.lemma == "не" or token.semantic_role in {"function_word", "pragmatic_marker"}:
            continue
        else:
            return None
    return ClauseThought(
        predicate_lexeme,
        tuple(dependencies),
        negated=any(t.lemma == "не" for t in tokens),
        confidence=0.86,
        predicate_class=predicate_class,
    )


def _parts(text: str) -> tuple[str, ...]:
    # Split only conjunctions/commas whose two sides independently parse.
    pieces = tuple(
        part.strip(" ,.;") for part in re.split(r",|;|\bи\b", text) if part.strip(" ,.;")
    )
    return pieces if len(pieces) > 1 else (text,)


def _group(
    text: str, perspective: str, antecedent: ClauseThought | None = None
) -> tuple[ClauseThought, ...]:
    fallback = subject_of(antecedent) if antecedent else None
    result = []
    for part in _parts(text):
        thought = parse_proposition(part, perspective=perspective, fallback_subject=fallback)
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
        (kind, tuple(asdict(t) for t in premises), tuple(asdict(t) for t in conclusions)),
        ensure_ascii=False,
        sort_keys=True,
    )
    meaning_id = hashlib.sha256((source + "|" + key).encode()).hexdigest()
    return MeaningRecord(meaning_id, source, text, kind, premises, conclusions)


def extract_meanings(
    text: str, source_event_id: str, *, perspective: str = "user"
) -> tuple[MeaningRecord, ...]:
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
                personal = _PERSONS[personal] if perspective == "user" else personal
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
        thoughts = _group(sentence, perspective)
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
    )


def _argument_key(argument: ConceptArgument) -> tuple:
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


def proposition_key(thought: ClauseThought, *, signed: bool = True) -> tuple:
    return (
        thought.predicate.lemma,
        thought.predicate_class,
        dict(thought.predicate.features).get("tense", "present"),
        tuple(sorted((dep.role, _argument_key(dep.argument)) for dep in thought.dependencies)),
        thought.negated if signed else None,
    )


def contextual(thought: ClauseThought) -> bool:
    return any(dep.argument.head.lemma in _RELATIVE | _THIRD_PERSON for dep in thought.dependencies)


def _match(
    pattern: ClauseThought, actual: ClauseThought, bindings: dict[str, ConceptArgument]
) -> dict[str, ConceptArgument] | None:
    if (
        pattern.predicate.lemma != actual.predicate.lemma
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
                    proposition_key(thought), MeaningProof(thought, (record.source_event_id,))
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
