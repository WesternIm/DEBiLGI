"""Productive Russian NLG over semantic roles, not authored reply templates.

The input is a small dependency-like thought: a predicate, its arguments and
discourse features.  The realizer chooses a role order, inflects every lexeme
and assembles a sentence.  No complete response wording is stored here.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, replace

from debilgi.language.morphology import DEFAULT_MORPHOLOGY
from debilgi.language.structured_utterance import (
    PARSER_VERSION,
    SemanticScope,
    SemanticTime,
    StructuredUtterance,
)
from debilgi.language.valency import assess_predicate_valency, governed_object_case

_ARGUMENT_ROLES = frozenset(
    {
        "subject",
        "direct_object",
        "indirect_object",
        "predicate_complement",
        "action_complement",
        "manner",
        "cause",
        "location",
        "topic",
        "comparison",
    }
)
_CORE_ROLE_ORDER = (
    "subject",
    "negation",
    "predicate",
    "indirect_object",
    "direct_object",
    "predicate_complement",
    "action_complement",
    "manner",
    "cause",
    "location",
    "topic",
    "comparison",
)
_QUESTION_FOCUS_LEMMAS = frozenset(
    {
        "где",
        "зачем",
        "как",
        "какой",
        "когда",
        "кто",
        "куда",
        "откуда",
        "почему",
        "сколько",
        "что",
    }
)


@dataclass(frozen=True, slots=True)
class ConceptLexeme:
    lemma: str
    surface: str
    part_of_speech: str | None = None
    features: tuple[tuple[str, str], ...] = ()
    entity_kind: str | None = None
    participant_ref: str | None = None


@dataclass(frozen=True, slots=True)
class ConceptArgument:
    head: ConceptLexeme
    modifiers: tuple[ConceptLexeme, ...] = ()
    preposition: str | None = None
    grammatical_case: str | None = None
    postmodifiers: tuple[ConceptLexeme, ...] = ()


@dataclass(frozen=True, slots=True)
class SemanticDependency:
    """One argument of a thought.

    ``head`` is the tree link (stage 4.24): the role of the dependency whose word governs this
    argument, ``None`` for the clause predicate. The default keeps every historical thought, its
    ``proposition_key`` and its readers unchanged; only an opt-in reader (``parse_proposition(
    tree=True)``) or a planner that knows the attachment (the object of «быть» in «хотел бы быть
    собакой») fills it.
    """

    role: str
    argument: ConceptArgument
    head: str | None = None


def identity_graph(value) -> dict:
    """``asdict`` as it was before the ``head`` link existed: an unattached dependency
    (``head is None``, every historical thought) has no ``head`` key, so ids and keys hashed
    from the graph stay byte for byte what they were."""

    def clean(item):
        if isinstance(item, dict):
            return {k: clean(v) for k, v in item.items() if not (k == "head" and v is None)}
        if isinstance(item, tuple):
            return tuple(clean(v) for v in item)
        if isinstance(item, list):
            return [clean(v) for v in item]
        return item

    return clean(asdict(value))


@dataclass(frozen=True, slots=True)
class ClauseThought:
    """A proposition selected by reasoning before any surface wording exists."""

    predicate: ConceptLexeme
    dependencies: tuple[SemanticDependency, ...]
    clause_kind: str = "statement"
    negated: bool = False
    response_polarity: bool | None = None
    focus_role: str | None = None
    confidence: float = 0.0
    evidence: tuple[str, ...] = ()
    predicate_class: str = "lexical"
    semantic_time: SemanticTime = SemanticTime()
    scopes: tuple[SemanticScope, ...] = ()
    parser_version: str = PARSER_VERSION
    source_event_id: str | None = None


@dataclass(frozen=True, slots=True)
class DependencyPattern:
    """An observed order of abstract roles; raw words are deliberately absent."""

    roles: tuple[str, ...]
    clause_kind: str = "statement"
    confidence: float = 0.5
    evidence_count: int = 1
    source: str = "observed"
    construction_id: str | None = None
    predicate_class: str = "any"
    required_roles: tuple[str, ...] = ()
    optional_roles: tuple[str, ...] = ()
    focus_role: str | None = None
    negation_mode: str = "any"
    utility: float = 0.0

    @property
    def valid(self) -> bool:
        if self.predicate_class == "dialogue_act":
            return bool(self.roles) and self.roles == tuple(
                f"act:{i}" for i in range(len(self.roles)))
        allowed = {*_ARGUMENT_ROLES, "predicate", "negation"}
        role_set = set(self.roles)
        negation_index = self.roles.index("negation") if "negation" in role_set else None
        predicate_index = self.roles.index("predicate") if "predicate" in role_set else None
        return (
            bool(self.roles)
            and len(self.roles) == len(set(self.roles))
            and role_set <= allowed
            # Every runtime thought has an explicit subject.  A learned order
            # without either core anchor used to append the missing anchor at
            # the end and produced surfaces such as ``разбил зеркало я``.
            and {"subject", "predicate"} <= role_set
            # In the single-clause constructions represented here, ``не``
            # scopes over the finite predicate.  Detached negation belongs to
            # a different construction and must not be guessed from a noisy
            # lesson.
            and (
                negation_index is None
                or predicate_index is not None
                and negation_index + 1 == predicate_index
            )
            and self.clause_kind in {"statement", "question", "any"}
            and self.predicate_class
            in {
                "any",
                "lexical",
                "modal",
                "existential",
                "copular",
            }
            and set(self.required_roles) <= set(self.roles)
            and set(self.optional_roles) <= set(self.roles)
            and not (set(self.required_roles) & set(self.optional_roles))
            and (self.focus_role is None or self.focus_role in self.roles)
            and self.negation_mode in {"any", "required", "forbidden"}
            and not (self.negation_mode == "required" and "negation" not in role_set)
            and not (self.negation_mode == "forbidden" and "negation" in role_set)
            and -1.0 <= self.utility <= 1.0
        )


@dataclass(frozen=True, slots=True)
class ProductiveSurface:
    text: str
    role_order: tuple[str, ...]
    confidence: float
    source: str
    construction_id: str | None = None


def dependency_pattern_id(pattern: DependencyPattern) -> str:
    """Return a stable identity for a word-free construction."""

    material = "|".join(
        (
            pattern.clause_kind,
            pattern.predicate_class,
            ",".join(pattern.roles),
            ",".join(pattern.required_roles),
            ",".join(pattern.optional_roles),
            pattern.focus_role or "",
            pattern.negation_mode,
        )
    )
    return f"construction:{hashlib.sha256(material.encode('utf-8')).hexdigest()[:24]}"


def productive_pattern_curriculum() -> tuple[DependencyPattern, ...]:
    """Return safe word-free Russian constructions used before organic feedback.

    These are grammatical priors, not prompt/answer pairs.  They only describe
    dependency slots, so new predicates and objects are still chosen by the
    local reasoner at runtime.
    """

    specifications = (
        DependencyPattern(
            ("subject", "predicate", "direct_object"),
            clause_kind="statement",
            predicate_class="lexical",
            required_roles=("subject", "predicate"),
            optional_roles=("direct_object",),
            negation_mode="forbidden",
            confidence=0.97,
            evidence_count=16,
            source="productive_curriculum",
        ),
        DependencyPattern(
            ("subject", "negation", "predicate", "direct_object"),
            clause_kind="statement",
            predicate_class="lexical",
            required_roles=("subject", "predicate", "negation"),
            optional_roles=("direct_object",),
            negation_mode="required",
            confidence=0.98,
            evidence_count=16,
            source="productive_curriculum",
        ),
        DependencyPattern(
            ("direct_object", "subject", "predicate"),
            clause_kind="statement",
            predicate_class="lexical",
            required_roles=("subject", "predicate"),
            optional_roles=("direct_object",),
            focus_role="direct_object",
            negation_mode="forbidden",
            confidence=0.91,
            evidence_count=12,
            source="productive_curriculum",
        ),
        DependencyPattern(
            (
                "subject",
                "predicate",
                "action_complement",
                "direct_object",
            ),
            clause_kind="statement",
            predicate_class="modal",
            required_roles=("subject", "predicate", "action_complement"),
            optional_roles=("direct_object",),
            negation_mode="forbidden",
            confidence=0.97,
            evidence_count=16,
            source="productive_curriculum",
        ),
        DependencyPattern(
            (
                "subject",
                "negation",
                "predicate",
                "action_complement",
                "direct_object",
            ),
            clause_kind="statement",
            predicate_class="modal",
            required_roles=(
                "subject",
                "predicate",
                "negation",
                "action_complement",
            ),
            optional_roles=("direct_object",),
            negation_mode="required",
            confidence=0.98,
            evidence_count=16,
            source="productive_curriculum",
        ),
        DependencyPattern(
            ("subject", "predicate", "predicate_complement"),
            clause_kind="statement",
            predicate_class="copular",
            required_roles=("subject", "predicate", "predicate_complement"),
            negation_mode="forbidden",
            confidence=0.96,
            evidence_count=14,
            source="productive_curriculum",
        ),
        DependencyPattern(
            ("location", "predicate", "subject", "topic"),
            clause_kind="statement",
            predicate_class="existential",
            required_roles=("subject", "predicate"),
            optional_roles=("location", "topic"),
            negation_mode="forbidden",
            confidence=0.95,
            evidence_count=14,
            source="productive_curriculum",
        ),
        DependencyPattern(
            ("subject", "negation", "predicate", "topic"),
            clause_kind="statement",
            predicate_class="existential",
            required_roles=("subject", "predicate", "negation"),
            optional_roles=("topic",),
            negation_mode="required",
            confidence=0.96,
            evidence_count=14,
            source="productive_curriculum",
        ),
        DependencyPattern(
            ("subject", "predicate", "direct_object"),
            clause_kind="question",
            predicate_class="lexical",
            required_roles=("subject", "predicate"),
            optional_roles=("direct_object",),
            negation_mode="forbidden",
            confidence=0.95,
            evidence_count=14,
            source="productive_curriculum",
        ),
    )
    return tuple(
        replace(pattern, construction_id=dependency_pattern_id(pattern))
        for pattern in specifications
        if pattern.valid
    )


def unresolved_answer_thought(*, confidence: float = 0.62) -> ClauseThought:
    """Represent an honest answer gap as meaning instead of a fallback string."""

    return ClauseThought(
        predicate=ConceptLexeme(
            "знать",
            "знаю",
            "verb",
            (("tense", "present"), ("number", "singular"), ("person", "first")),
        ),
        dependencies=(
            SemanticDependency(
                "subject",
                ConceptArgument(
                    ConceptLexeme(
                        "я",
                        "я",
                        "pronoun",
                        (("number", "singular"), ("person", "first")),
                    )
                ),
            ),
            SemanticDependency(
                "direct_object",
                ConceptArgument(
                    ConceptLexeme("ответ", "ответ", "noun"),
                    grammatical_case="genitive",
                ),
            ),
            SemanticDependency(
                "topic",
                ConceptArgument(
                    ConceptLexeme(
                        "вопрос",
                        "вопрос",
                        "noun",
                        (("number", "singular"), ("gender", "masculine")),
                    ),
                    modifiers=(ConceptLexeme("этот", "этот", "adjective"),),
                    preposition="на",
                    grammatical_case="accusative",
                ),
            ),
        ),
        negated=True,
        confidence=confidence,
        evidence=("no_grounded_answer",),
    )


def observe_dependency_patterns(
    structure: StructuredUtterance,
) -> tuple[DependencyPattern, ...]:
    """Abstract locally parsed sentences into reusable, word-free role orders."""

    patterns: list[DependencyPattern] = []
    seen: set[tuple[str, ...]] = set()
    for clause in structure.clauses:
        if clause.predicate_index is None or not clause.subject_indices:
            continue
        positioned: list[tuple[int, str]] = [
            (min(clause.subject_indices), "subject"),
            (clause.predicate_index, "predicate"),
        ]
        for indices, role in (
            (clause.action_complement_indices, "action_complement"),
            (clause.direct_object_indices, "direct_object"),
            (clause.indirect_object_indices, "indirect_object"),
            (clause.predicate_complement_indices, "predicate_complement"),
        ):
            if indices:
                positioned.append((min(indices), role))
        if clause.negated:
            negation_index = next(
                (
                    token.index
                    for token in structure.tokens[clause.token_start : clause.token_end]
                    if token.lemma == "не"
                ),
                clause.predicate_index - 1,
            )
            positioned.append((negation_index, "negation"))
        roles = tuple(
            dict.fromkeys(role for _, role in sorted(positioned, key=lambda item: item[0]))
        )
        clause_kind = (
            clause.clause_kind if clause.clause_kind in {"statement", "question"} else "any"
        )
        predicate_class = _observed_predicate_class(structure, clause.predicate_index, clause)
        focus_role = _observed_focus_role(clause)
        required_roles = tuple(role for role in roles if role in {"subject", "predicate"})
        optional_roles = tuple(role for role in roles if role not in required_roles)
        pattern = DependencyPattern(
            roles=roles,
            clause_kind=clause_kind,
            confidence=clause.confidence,
            source="local_parse",
            predicate_class=predicate_class,
            required_roles=required_roles,
            optional_roles=optional_roles,
            focus_role=focus_role,
            negation_mode="required" if clause.negated else "forbidden",
        )
        pattern = replace(pattern, construction_id=dependency_pattern_id(pattern))
        if pattern.valid and pattern.roles not in seen:
            seen.add(pattern.roles)
            patterns.append(pattern)
    return tuple(patterns)


def realize_argument(
    argument: ConceptArgument, *, predicate_lemma: str = "", role: str = "direct_object"
) -> str:
    """Render one selected semantic argument for a focused answer."""
    case = _case_for_role(predicate_lemma, role, argument)
    return _render_argument(argument, case=case)


def realize_thought(
    thought: ClauseThought,
    patterns: tuple[DependencyPattern, ...] = (),
    *,
    limit: int = 6,
) -> tuple[ProductiveSurface, ...]:
    """Realize several grammatical role orders without retrieving a reply.

    With ``DEBILGI_GRAMMARIAN=1`` the personal network of ``language/grammarian.py`` words the
    plan first; when it refuses (low confidence, an incomplete or unsupported plan, an error) —
    and always when the flag is off — the deterministic realizer below answers exactly as
    before. The network's surface, when accepted, stands first and the deterministic ones
    follow as alternatives for the reader check of the composer.
    """
    surfaces = _realize_thought_deterministic(thought, patterns, limit=limit)
    from debilgi.language import grammarian

    if not grammarian.enabled() or not surfaces or thought.predicate_class == "dialogue_act":
        return surfaces
    if len(patterns) == 1:
        # one explicit pattern is an order the caller demands (an embedded question, a short
        # act): not the network's to choose
        grammarian.note_fallback(thought, "order_fixed_by_caller")
        return surfaces
    answer, _record = grammarian.realize(thought)
    if answer is None:
        return surfaces
    text = _close_sentence(thought, answer.body)
    if not text:
        return surfaces
    seen = " ".join(text.casefold().split())
    network = ProductiveSurface(
        text=text,
        role_order=answer.role_order,
        confidence=min(thought.confidence, answer.confidence),
        source="grammarian",
        construction_id=None,
    )
    rest = tuple(s for s in surfaces if " ".join(s.text.casefold().split()) != seen)
    return (network, *rest)[: max(1, min(limit, 12))]


def _close_sentence(thought: ClauseThought, body: str) -> str:
    """The answer marker, the capital and the sentence end of a clause (shared by realizers)."""
    if thought.response_polarity is not None:
        marker = "Да" if thought.response_polarity else "Нет"
        body = f"{marker}, {body}"
    punctuation = "?" if thought.clause_kind == "question" else "."
    body = body.strip(" .!?…")
    return body[:1].upper() + body[1:] + punctuation if body else ""


def _thought_content() -> bool:
    from debilgi.config import thought_content_enabled

    return thought_content_enabled()


def _realize_thought_deterministic(
    thought: ClauseThought,
    patterns: tuple[DependencyPattern, ...] = (),
    *,
    limit: int = 6,
) -> tuple[ProductiveSurface, ...]:
    if thought.predicate_class == "dialogue_act":
        import re

        order = tuple(d.role for d in thought.dependencies)
        pattern = next((p for p in patterns if p.valid and p.predicate_class == "dialogue_act"
                        and p.roles == order), None)
        if pattern is None:
            return ()
        words = []
        for item in thought.dependencies:
            if not item.role.startswith("act:"):
                return ()
            lexeme = item.argument.head
            features = dict(lexeme.features)
            if features.get("number") == "plural":
                features.pop("gender", None)
            word = (DEFAULT_MORPHOLOGY.inflect(lexeme.lemma, **features)
                    if features else lexeme.surface)
            if not word:
                return ()
            words.append(word)
        text = re.sub(r"\s+([,.!?;:)])", r"\1", " ".join(words))
        return ((ProductiveSurface(text, tuple(d.role for d in thought.dependencies),
                                   thought.confidence, "dialogue_act", None),) if text else ())
    role_values = {item.role: item.argument for item in thought.dependencies}
    if "subject" not in role_values or not thought.predicate.lemma:
        return ()
    predicate_roles = set(role_values)
    if (
        (thought.predicate.lemma in {"мочь", "уметь", "хотеть"}
         or thought.predicate_class == "modal" and _thought_content())
        and "action_complement" in predicate_roles
    ):
        # Arguments of the embedded action aren't arguments of the modal.
        predicate_roles = {"subject", "action_complement"}
    heads = {item.role: item.head for item in thought.dependencies if item.head}
    # A comparison and the arguments of an embedded verb are not arguments of the predicate.
    predicate_roles -= {*heads, "comparison"} - {"action_complement"}
    valency = assess_predicate_valency(
        thought.predicate.lemma,
        present_roles=frozenset(predicate_roles),
        complement_roles=frozenset(
            role_name
            for role_name in role_values
            if role_name in {"predicate_complement", "action_complement", "location", "topic"}
        ),
    )
    if valency.missing_roles or valency.unexpected_roles:
        return ()
    present_roles = {
        *role_values,
        "predicate",
        *(("negation",) if thought.negated else ()),
    }
    orders: list[tuple[tuple[str, ...], float, str, str | None]] = []
    for pattern in patterns:
        if (
            not pattern.valid
            or pattern.clause_kind not in {"any", thought.clause_kind}
            or pattern.predicate_class not in {"any", thought.predicate_class}
            or (pattern.negation_mode == "required" and not thought.negated)
            or (pattern.negation_mode == "forbidden" and thought.negated)
            # Never insert negation into an arbitrary learned order.  That was
            # the source of ``зеркало разбил я не`` and also changed modal
            # scope from ``не могу сделать`` to ``могу не сделать``.
            or (("negation" in pattern.roles) != thought.negated)
            or not set(pattern.required_roles) <= present_roles
        ):
            continue
        ordered = tuple(role for role in pattern.roles if role in present_roles)
        missing = tuple(
            role for role in _CORE_ROLE_ORDER if role in present_roles and role not in ordered
        )
        complete = (*ordered, *missing)
        if set(complete) == present_roles:
            support = min(1.0, pattern.evidence_count / 8.0)
            contextual_fit = 1.0
            if pattern.focus_role is not None:
                contextual_fit = 1.0 if pattern.focus_role == thought.focus_role else 0.72
            learned_score = (
                pattern.confidence * 0.65
                + support * 0.20
                + contextual_fit * 0.15
                + pattern.utility * 0.05
            )
            orders.append(
                (
                    complete,
                    learned_score,
                    pattern.source,
                    pattern.construction_id,
                )
            )

    base_order = (
        ("location", "predicate", "subject", "topic")
        if thought.predicate_class == "existential"
        else (
            "subject",
            "negation",
            "predicate",
            "action_complement",
            "predicate_complement",
            "direct_object",
            "indirect_object",
            "location",
            "topic",
            "manner",
            "comparison",
        )
        if thought.predicate_class == "modal"
        else _CORE_ROLE_ORDER
    )
    default = tuple(role for role in base_order if role in present_roles)
    orders.append((default, 0.86, "productive_grammar", None))
    if (thought.focus_role in present_roles and thought.focus_role != "predicate"
            and (thought.focus_role != "subject" or thought.clause_kind == "question")):
        focused = (
            thought.focus_role,
            *(role for role in default if role != thought.focus_role),
        )
        orders.append(
            (
                focused,
                0.94 if thought.clause_kind == "question" else 0.68,
                "information_structure",
                None,
            )
        )
    if heads:
        # An argument of an embedded verb stands next to that verb, whatever order a learned
        # pattern gave the roles of the whole clause («хотел бы собаку быть» is not a clause).
        orders = [
            (_attach_to_heads(order, heads, thought.focus_role), *rest)
            for order, *rest in orders
        ]
    focused_argument = (
        role_values.get(thought.focus_role) if thought.focus_role is not None else None
    )
    requires_fronted_question_focus = bool(
        thought.clause_kind == "question"
        and thought.focus_role in present_roles
        and focused_argument is not None
        and any(lexeme.lemma in _QUESTION_FOCUS_LEMMAS
                for lexeme in (focused_argument.head, *focused_argument.modifiers))
    )
    if requires_fronted_question_focus:
        # ``focus_role`` is an explicit information-structure instruction.
        # A learned order may not demote its WH argument into an unlicensed
        # in-situ position such as ``думаешь ты что о ...``.
        orders = [item for item in orders if item[0] and item[0][0] == thought.focus_role]

    surfaces: list[ProductiveSurface] = []
    seen_orders: set[tuple[str, ...]] = set()
    seen_texts: set[str] = set()
    for order, order_confidence, source, construction_id in sorted(
        orders, key=lambda item: item[1], reverse=True
    ):
        if order in seen_orders:
            continue
        seen_orders.add(order)
        text = _realize_order(thought, order, role_values)
        normalized = " ".join(text.casefold().split())
        if not text or normalized in seen_texts:
            continue
        seen_texts.add(normalized)
        surfaces.append(
            ProductiveSurface(
                text=text,
                role_order=order,
                confidence=min(thought.confidence, order_confidence),
                source=source,
                construction_id=construction_id,
            )
        )
        if len(surfaces) >= max(1, min(limit, 12)):
            break
    return tuple(surfaces)


def _attach_to_heads(
    order: tuple[str, ...], heads: dict[str, str], focus_role: str | None = None
) -> tuple[str, ...]:
    """Move every role with a ``head`` right behind that head role (stable, no role lost)."""
    movable = [
        role for role in order
        if role in heads and heads[role] in order and not (order[0] == role == focus_role)
    ]
    result = [role for role in order if role not in movable]
    for role in movable:
        head = heads[role]
        if head not in result:
            result.append(role)
            continue
        index = result.index(head) + 1
        while index < len(result) and heads.get(result[index]) == head:
            index += 1
        result.insert(index, role)
    return tuple(result)


def _observed_predicate_class(
    structure: StructuredUtterance,
    predicate_index: int,
    clause: object,
) -> str:
    token = structure.tokens[predicate_index]
    if getattr(clause, "modal_lemmas", ()):
        return "modal"
    if token.lemma in {"быть", "иметься", "существовать"} or token.normalized == "есть":
        return "existential"
    if token.part_of_speech in {"noun", "adjective", "participle"}:
        return "copular"
    return "lexical"


def _observed_focus_role(clause: object) -> str | None:
    target = getattr(clause, "question_target_index", None)
    if target is None:
        return None
    for indices, role in (
        (getattr(clause, "subject_indices", ()), "subject"),
        (getattr(clause, "direct_object_indices", ()), "direct_object"),
        (getattr(clause, "indirect_object_indices", ()), "indirect_object"),
        (
            getattr(clause, "predicate_complement_indices", ()),
            "predicate_complement",
        ),
    ):
        if target in indices:
            return role
    return "predicate" if target == getattr(clause, "predicate_index", None) else None


def _realize_order(
    thought: ClauseThought,
    order: tuple[str, ...],
    role_values: dict[str, ConceptArgument],
) -> str:
    subject = role_values.get("subject")
    heads = {item.role: item.head for item in thought.dependencies if item.head}
    predicate_features = _predicate_features(thought.predicate, subject)
    conditional = dict(thought.predicate.features).get("mood") == "conditional"
    words: list[str] = []
    for role in order:
        if role == "negation":
            words.append("не")
        elif role == "predicate":
            if (
                thought.predicate_class == "copular"
                and thought.predicate.lemma == "быть"
                and predicate_features.get("tense", "present") == "present"
            ):
                continue
            predicate = _realize_predicate(
                thought.predicate,
                predicate_features,
            )
            words.append(predicate)
            if conditional and predicate:
                # Conditional mood: the past form plus the particle after the verb.
                words.append("бы")
        else:
            argument = role_values.get(role)
            if argument is None:
                continue
            if role == "subject" and ("elided", "true") in argument.head.features:
                continue  # the person is carried by the verb form (pro-drop), not said
            governor = role_values.get(heads.get(role, ""))
            case = (
                _case_for_role(governor.head.lemma, role, argument, nested=True)
                if governor is not None
                else _case_for_role(thought.predicate.lemma, role, argument)
            )
            if role == "comparison" and not argument.preposition:
                # The relation of likeness is a meaning of the plan; its closed-class word
                # is the one the plan names, «как» when it names none.
                argument = replace(argument, preposition="как")
            if (
                thought.predicate_class in {"copular", "modal"}
                and role == "predicate_complement"
                and subject
                and argument.head.part_of_speech
                in {
                    "adjective",
                    "participle",
                }
            ):
                agreement = dict(argument.head.features)
                subject_features = _inferred_head_features(subject.head)
                for name in ("number", "gender"):
                    if subject_features.get(name):
                        agreement[name] = subject_features[name]
                argument = replace(
                    argument, head=replace(argument.head, features=tuple(agreement.items()))
                )
            rendered = _render_argument(
                argument,
                case=case,
                predicative=(
                    role == "predicate_complement" and thought.predicate_class == "copular"
                ),
            )
            if not rendered:
                return ""
            words.append(rendered)
    if not words:
        return ""
    return _close_sentence(thought, " ".join(words))


def _realize_predicate(
    predicate: ConceptLexeme,
    features: dict[str, str],
) -> str:
    if predicate.part_of_speech != "verb":
        return predicate.surface
    inflected = DEFAULT_MORPHOLOGY.inflect(predicate.lemma, **features)
    if inflected:
        return inflected
    reading = DEFAULT_MORPHOLOGY.best(predicate.lemma)
    if (
        features.get("tense") == "future"
        and reading is not None
        and reading.aspect == "imperfective"
    ):
        auxiliary = DEFAULT_MORPHOLOGY.inflect(
            "быть",
            tense="future",
            person=features.get("person", "third"),
            number=features.get("number", "singular"),
        )
        if auxiliary:
            return f"{auxiliary} {reading.lemma}"
    return predicate.surface


def _predicate_features(
    predicate: ConceptLexeme,
    subject: ConceptArgument | None,
) -> dict[str, str]:
    features = {
        name: value
        for name, value in predicate.features
        if name in {"tense", "number", "gender", "person"}
    }
    if dict(predicate.features).get("mood") == "conditional":
        features["tense"] = "past"
    if subject is None:
        return features
    subject_features = _inferred_head_features(subject.head)
    if subject.head.lemma == "я":
        features["number"] = "singular"
        if features.get("tense") != "past":
            features["person"] = "first"
    elif subject.head.lemma == "ты":
        features["number"] = "singular"
        if features.get("tense") != "past":
            features["person"] = "second"
    elif subject.head.lemma == "мы":
        features["number"] = "plural"
        if features.get("tense") != "past":
            features["person"] = "first"
    elif subject.head.lemma == "вы":
        features["number"] = "plural"
        if features.get("tense") != "past":
            features["person"] = "second"
    elif features.get("tense") != "past":
        features["person"] = "third"
    if features.get("tense") == "past":
        features.pop("person", None)
        for name in ("number", "gender"):
            if subject_features.get(name):
                features[name] = subject_features[name]
    else:
        # Russian present/future forms agree in person and number.  A gender
        # grammeme is incompatible here and can make the morphology backend
        # fall back to a gerund such as ``приходя``.
        features.pop("gender", None)
        for name in ("person", "number"):
            if subject_features.get(name):
                features[name] = subject_features[name]
    return features


def _nominal_complement_case(verb_lemma: str) -> str | None:
    """The case a bare noun takes after a verb that cannot govern the accusative.

    An intransitive verb with no lexical profile (быть, стать, казаться, остаться … and any
    verb not seen before) can only carry its noun as a predicative complement, which Russian
    puts in the instrumental. The verb's transitivity comes from its morphology, not from a
    list of verbs; a lexical profile of government (``valency``) always goes first.
    """
    reading = DEFAULT_MORPHOLOGY.best(verb_lemma)
    if reading is not None and reading.features().get("transitivity") == "intransitive":
        return "instrumental"
    return None


def _case_for_role(
    predicate_lemma: str,
    role: str,
    argument: ConceptArgument,
    *,
    nested: bool = False,
) -> str | None:
    """Case of an argument; ``predicate_lemma`` is the word that governs it (its ``head``)."""
    if argument.grammatical_case:
        return argument.grammatical_case
    if role in {"subject", "comparison"}:
        return "nominative"
    if role == "direct_object":
        governed = governed_object_case(predicate_lemma, role, "")
        if governed:
            return governed
        return (nested and _nominal_complement_case(predicate_lemma)) or "accusative"
    if role == "indirect_object":
        return governed_object_case(predicate_lemma, role, "dative")
    if role == "predicate_complement":
        return "instrumental"
    if role == "action_complement":
        return None
    if argument.preposition:
        expected = DEFAULT_MORPHOLOGY.expected_cases(argument.preposition)
        return (
            sorted(expected)[0] if len(expected) == 1 else dict(argument.head.features).get("case")
        )
    return dict(argument.head.features).get("case")


def _render_argument(
    argument: ConceptArgument, *, case: str | None, predicative: bool = False
) -> str:
    head_features = _inferred_head_features(argument.head)
    requested: dict[str, str] = {}
    if case:
        requested["case"] = case
    if head_features.get("number"):
        requested["number"] = head_features["number"]
    if argument.head.part_of_speech in {"adjective", "participle"} and head_features.get("gender"):
        requested["gender"] = head_features["gender"]
    inflection_base = argument.head.lemma
    name_reading = DEFAULT_MORPHOLOGY.best(inflection_base)
    proper_name = bool(name_reading and name_reading.is_known
                       and {"Name", "Surn"} & name_reading.grammemes
                       and not argument.head.surface.startswith("«"))
    if proper_name:
        inflection_base = inflection_base[:1].upper() + inflection_base[1:]
    if argument.head.part_of_speech == "participle":
        # Its lemma is a verb; inflect the observed participle to retain voice.
        inflection_base = argument.head.surface
        reading = DEFAULT_MORPHOLOGY.best(inflection_base)
        if reading and "PRTS" in reading.grammemes:
            requested.pop("case", None)
    if predicative and argument.head.part_of_speech in {"participle", "adjective"}:
        short = {k: v for k, v in requested.items() if k != "case"}
        short["form"] = (
            "short_participle"
            if argument.head.part_of_speech == "participle"
            else "short_adjective"
        )
        short_surface = DEFAULT_MORPHOLOGY.inflect(argument.head.surface, **short)
        original = DEFAULT_MORPHOLOGY.best(argument.head.surface)
        short_reading = DEFAULT_MORPHOLOGY.best(short_surface) if short_surface else None
        if short_surface and (
            argument.head.part_of_speech == "participle"
            or dict(argument.head.features).get("case") == "instrumental"
            or original
            and "ADJS" in original.grammemes
            or short_reading
            and short_reading.part_of_speech == "participle"
        ):
            return short_surface
    head = (
        argument.head.surface
        if (argument.head.entity_kind is not None and not proper_name
            or head_features.get("indeclinable") == "true")
        else DEFAULT_MORPHOLOGY.inflect(inflection_base, **requested) or argument.head.surface
    )
    modifier_features = dict(requested)
    for name in ("number", "gender"):
        if head_features.get(name):
            modifier_features[name] = head_features[name]
    # Animacy distinguishes accusative adjective/participle forms.  Passing
    # it together with instrumental or prepositional grammemes makes
    # pymorphy reject an otherwise valid inflection and leaks the nominative
    # modifier (``обучающийся собеседником``).
    if (
        case == "accusative"
        and head_features.get("animacy")
        and (head_features.get("number") == "plural" or head_features.get("gender") == "masculine")
    ):
        modifier_features["animacy"] = head_features["animacy"]
    modifiers = tuple(
        DEFAULT_MORPHOLOGY.inflect(item.lemma, **modifier_features) or item.surface
        for item in argument.modifiers
    )
    postmodifiers = tuple(item.surface for item in argument.postmodifiers)
    phrase = " ".join((*modifiers, head, *postmodifiers)).strip()
    preposition = preposition_allomorph(argument.preposition, phrase)
    return f"{preposition} {phrase}" if preposition else phrase


def preposition_allomorph(preposition: str | None, next_word: str) -> str | None:
    """«о» stands as «об» before a vowel (shared by every realizer)."""
    if preposition == "о" and next_word[:1].casefold() in "аэиоуы":
        return "об"
    return preposition


def _inferred_head_features(head: ConceptLexeme) -> dict[str, str]:
    """Combine explicit semantic features with the noun's lexical agreement data.

    Reasoners are allowed to construct a thought from a bare lemma.  In that case
    the surface realizer still needs gender, number and animacy to inflect attached
    adjectives.  Explicit features remain authoritative because they can express
    a context-specific plural or grammatical form.
    """

    inferred: dict[str, str] = {}
    reading = DEFAULT_MORPHOLOGY.best(head.lemma)
    if reading is not None:
        inferred.update(reading.features())
    inferred.update(dict(head.features))
    if head.lemma == "я" and not inferred.get("gender") and _thought_content():
        # (4.42, flag only) the agent speaks of itself in the gender of its own character (the
        # owner's decision, the one recorded for the grammarian): never in the gender that an
        # ambiguous reading of the adjective («тупой» is also feminine genitive) would lend
        from debilgi.language.grammarian import self_gender

        inferred["gender"] = self_gender()
    if inferred.get("number") == "plural":
        inferred.pop("gender", None)
    return inferred
