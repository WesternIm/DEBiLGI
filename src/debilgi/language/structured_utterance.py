"""Shadow structural analysis between word understanding and response planning.

The structures in this module describe meaning-bearing roles.  They do not
render replies and therefore cannot turn a niche case into a hard-coded answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from debilgi.language.addressing import DEFAULT_ADDRESSING
from debilgi.language.character_markov import normalize_word
from debilgi.language.entities import EntityAnalysis, EntityMention
from debilgi.language.function_words import service_word_reason
from debilgi.language.grammar import role
from debilgi.language.lexicon import WordAnalysis
from debilgi.language.morphology import DEFAULT_MORPHOLOGY, MorphologicalReading
from debilgi.language.pragmatics import analyze_pragmatics
from debilgi.language.tokenizer import normalize_chat_token
from debilgi.language.valency import predicate_valency

_CLAUSE_END = frozenset(".!?;…")
_LOCATION_PREPOSITIONS = frozenset(
    {"в", "во", "на", "над", "под", "за", "около", "возле", "внутри", "между"}
)
_SOURCE_PREPOSITIONS = frozenset({"из", "от", "с", "со"})
_DESTINATION_PREPOSITIONS = frozenset({"к", "ко", "в", "во", "на"})
_TOPIC_PREPOSITIONS = frozenset({"о", "об", "обо", "про"})
_INSTRUMENT_PREPOSITIONS = frozenset({"с", "со"})
_MODAL_LEMMAS = frozenset(
    {"мочь", "хотеть", "должный", "следовать", "надо", "нужно", "можно", "нельзя"}
)
_EMOTIONAL_INTENSIFIERS = frozenset({"бля", "блять", "блин", "нахуй", "сука", "черт", "чёрт"})
_ATTENTION_CALL_RE = re.compile(r"^(?:а+у+|ал+о+|ал+е+|эй+)[?!.…]*$", re.IGNORECASE)
_QUOTE_TOKENS = frozenset({"«", "»", "“", "”", "„", '"'})
_COORDINATION_RELATIONS = {
    "и": "additive",
    "или": "alternative",
    "либо": "alternative",
    "а": "contrast",
    "но": "contrast",
    "зато": "contrast",
}
_COORDINATABLE_PARTS = frozenset(
    {"noun", "pronoun", "adjective", "participle", "number", "adverb"}
)
_SUBORDINATE_MARKERS = {
    "что": "ccomp",
    "чтобы": "ccomp:purpose",
    "если": "advcl:condition",
    "когда": "advcl:time",
    "пока": "advcl:time",
    "потому": "advcl:cause",
    "хотя": "advcl:concession",
}


@dataclass(frozen=True, slots=True)
class TokenInterpretation:
    index: int
    clause_index: int | None
    surface: str
    normalized: str
    corrected_form: str | None
    lemma: str
    part_of_speech: str | None
    grammatical_features: tuple[tuple[str, str], ...]
    grammemes: tuple[str, ...]
    semantic_role: str
    entity_kind: str | None = None
    entity_role: str | None = None
    entity_id: int | None = None
    pragmatic_functions: tuple[str, ...] = ()
    selected_meaning: str | None = None
    personal_meaning: str | None = None
    local_meaning: str | None = None
    meaning_candidates: tuple[str, ...] = ()
    style_labels: tuple[str, ...] = ()
    lexical_source: str | None = None
    meaning_confidence: float = 0.0
    context_fit: float = 0.0
    unknown_kind: str | None = None
    confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class ClauseComplement:
    role: str
    preposition_index: int
    object_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ClauseCoordination:
    """A structural operator connecting equal semantic members."""

    relation: str
    conjunction_index: int
    member_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DependencyArc:
    """A local dependency edge between two overt tokens.

    Clause frames remain the semantic API used by the reasoners.  These arcs
    expose the same analysis as a dependency graph so diagnostics and learned
    construction patterns do not have to reconstruct word-to-word links from
    flat POS tags.
    """

    clause_index: int
    head_index: int
    dependent_index: int
    relation: str


@dataclass(frozen=True, slots=True)
class ClauseLink:
    """A relation between matrix, subordinate or coordinated clauses."""

    parent_clause_index: int
    child_clause_index: int
    relation: str
    marker_index: int


@dataclass(frozen=True, slots=True)
class ClauseFrame:
    clause_index: int
    token_start: int
    token_end: int
    clause_kind: str
    subject_indices: tuple[int, ...]
    predicate_index: int | None
    action_complement_indices: tuple[int, ...]
    predicate_complement_indices: tuple[int, ...]
    direct_object_indices: tuple[int, ...]
    indirect_object_indices: tuple[int, ...]
    complements: tuple[ClauseComplement, ...]
    coordinations: tuple[ClauseCoordination, ...]
    vocative_indices: tuple[int, ...]
    question_target_index: int | None
    negated: bool
    modal_lemmas: tuple[str, ...]
    implicit_subject: str | None
    implicit_predicate: str | None
    elliptical: bool
    confidence: float


@dataclass(frozen=True, slots=True)
class StructuredUtterance:
    text: str
    tokens: tuple[TokenInterpretation, ...]
    clauses: tuple[ClauseFrame, ...]
    dependencies: tuple[DependencyArc, ...]
    clause_links: tuple[ClauseLink, ...]
    niche_signals: tuple[str, ...]
    ambiguities: tuple[str, ...]
    confidence: float


def analyze_structured_utterance(
    text: str,
    lexical_analyses: tuple[WordAnalysis, ...] = (),
    entity_analysis: EntityAnalysis | None = None,
) -> StructuredUtterance:
    """Build token cards and clause roles without affecting response selection."""

    sentence = DEFAULT_MORPHOLOGY.analyze_sentence(text)
    addressing = DEFAULT_ADDRESSING.analyze(text)
    pragmatic = analyze_pragmatics(text)
    entity_analysis = entity_analysis or EntityAnalysis((), (), ())
    lexical_by_form = _lexical_index(lexical_analyses)
    address_indices = frozenset(
        index for span in addressing.addresses for index in range(span.token_start, span.token_end)
    )
    entity_by_index = _entities_by_index(sentence.tokens, entity_analysis.mentions)
    pragmatic_by_index = _pragmatics_by_index(sentence.tokens, pragmatic.cues)
    quoted_indices = _quoted_indices(sentence.tokens)

    token_cards: list[TokenInterpretation] = []
    ambiguities: list[str] = []
    niche_signals: list[str] = []
    for index, (surface, reading, alternatives) in enumerate(
        zip(sentence.tokens, sentence.readings, sentence.alternatives, strict=True)
    ):
        entity = entity_by_index.get(index)
        corrected = reading.corrected_form if reading is not None and entity is None else None
        expressive = normalize_chat_token(surface)
        normalized = corrected or normalize_word(surface)
        lexical = lexical_by_form.get(normalized)
        lemma = (
            entity.canonical
            if entity is not None
            else lexical.lemma
            if lexical is not None
            and lexical.lemma
            and lexical.source in {"server_slang", "personal_lexicon"}
            else reading.lemma
            if reading is not None
            else lexical.lemma
            if lexical is not None and lexical.lemma
            else normalized
        )
        part = reading.part_of_speech if reading is not None else _normalized_part(lexical)
        features = (
            reading.features()
            if reading is not None
            else lexical.grammatical_features
            if lexical
            else {}
        )
        functions = pragmatic_by_index.get(index, ())
        service_reason = service_word_reason(
            lemma,
            part_of_speech=part,
            question_context=text.rstrip().endswith("?"),
        )
        initial_role = (
            "vocative"
            if index in address_indices
            else "quoted_content"
            if index in quoted_indices and surface not in _QUOTE_TOKENS
            else "punctuation"
            if not surface[:1].isalnum()
            else "pragmatic_marker"
            if normalized in _EMOTIONAL_INTENSIFIERS or functions
            else "function_word"
            if service_reason is not None
            else "unassigned_content"
        )
        selected_meaning = _selected_meaning(lexical)
        candidates = lexical.dictionary_meanings if lexical is not None else ()
        unknown_kind = lexical.unknown_kind.value if lexical and lexical.unknown_kind else None
        confidence = _token_confidence(reading, lexical, entity)
        token_cards.append(
            TokenInterpretation(
                index=index,
                clause_index=None,
                surface=surface,
                normalized=normalized,
                corrected_form=corrected,
                lemma=lemma,
                part_of_speech=part,
                grammatical_features=tuple(sorted(features.items())),
                grammemes=tuple(sorted(reading.grammemes)) if reading is not None else (),
                semantic_role=initial_role,
                entity_kind=entity.kind if entity is not None else None,
                entity_role=entity.role if entity is not None else None,
                entity_id=entity.entity_id if entity is not None else None,
                pragmatic_functions=functions,
                selected_meaning=selected_meaning,
                personal_meaning=(lexical.personal_interpretation if lexical else None),
                local_meaning=lexical.local_meaning if lexical else None,
                meaning_candidates=candidates,
                style_labels=lexical.style_labels if lexical else (),
                lexical_source=lexical.source if lexical is not None else None,
                meaning_confidence=(lexical.confidence.meaning if lexical else 0.0),
                context_fit=lexical.confidence.context_fit if lexical else 0.0,
                unknown_kind=unknown_kind,
                confidence=confidence,
            )
        )
        if corrected and expressive.kind is not None:
            niche_signals.append(
                f"expressive_{expressive.kind}:{surface}->{corrected}"
            )
            niche_signals.append(
                f"expressive_intensity:{expressive.intensity:.2f}"
            )
        elif corrected:
            niche_signals.append(f"spelling_correction:{surface}->{corrected}")
        if entity is not None:
            niche_signals.append(f"entity:{entity.kind}")
        if lexical is not None and lexical.source in {"server_slang", "personal_lexicon"}:
            niche_signals.append(f"contextual_lexicon:{lexical.source}")
        if lexical is not None and lexical.is_neologism:
            niche_signals.append("neologism")
        if normalized in _EMOTIONAL_INTENSIFIERS:
            niche_signals.append("emotional_intensifier")
        if lexical is not None and lexical.unknown_kind is not None:
            ambiguities.append(f"{surface}:{lexical.unknown_kind.value}")
        elif len(candidates) > 1 and selected_meaning is None:
            ambiguities.append(f"{surface}:unresolved_sense")
        if len(alternatives) > 1 and reading is not None:
            distinct = {(item.lemma, item.part_of_speech) for item in alternatives}
            if len(distinct) > 1:
                ambiguities.append(f"{surface}:morphology_resolved_as:{reading.lemma}")

    normalized_text = " ".join(text.casefold().replace("ё", "е").split())
    if _ATTENTION_CALL_RE.fullmatch(normalized_text):
        niche_signals.append("attention_call")
    niche_signals.extend(
        f"pragmatic:{function}"
        for function in sorted(pragmatic.functions)
        if function not in {"turn_management"}
    )
    if address_indices:
        niche_signals.append("direct_address")

    clauses: list[ClauseFrame] = []
    mutable_cards = token_cards
    for clause_index, (start, end) in enumerate(
        _clause_ranges(sentence.tokens, tuple(token_cards))
    ):
        frame = _analyze_clause(
            clause_index,
            start,
            end,
            tuple(mutable_cards),
            address_indices,
            text.rstrip().endswith("?"),
        )
        clauses.append(frame)
        role_updates = _roles_for_frame(frame)
        for index in range(start, end):
            card = mutable_cards[index]
            role_name = role_updates.get(index, card.semantic_role)
            mutable_cards[index] = replace(
                card,
                clause_index=clause_index,
                semantic_role=role_name,
            )
        if frame.elliptical:
            niche_signals.append("ellipsis")
        if frame.implicit_predicate:
            niche_signals.append(f"implicit_predicate:{frame.implicit_predicate}")

    lexical_cards = tuple(card for card in mutable_cards if card.surface[:1].isalnum())
    token_confidence = (
        sum(card.confidence for card in lexical_cards) / len(lexical_cards)
        if lexical_cards
        else 0.0
    )
    clause_confidence = (
        sum(frame.confidence for frame in clauses) / len(clauses) if clauses else 0.0
    )
    dependencies = _dependency_arcs(tuple(mutable_cards), tuple(clauses))
    clause_links = _clause_links(tuple(mutable_cards), tuple(clauses))
    return StructuredUtterance(
        text=text,
        tokens=tuple(mutable_cards),
        clauses=tuple(clauses),
        dependencies=dependencies,
        clause_links=clause_links,
        niche_signals=tuple(dict.fromkeys(niche_signals)),
        ambiguities=tuple(dict.fromkeys(ambiguities)),
        confidence=min(1.0, token_confidence * 0.45 + clause_confidence * 0.55),
    )


def _analyze_clause(
    clause_index: int,
    start: int,
    end: int,
    cards: tuple[TokenInterpretation, ...],
    address_indices: frozenset[int],
    utterance_is_question: bool,
) -> ClauseFrame:
    # Structural operators must survive long enough to connect content words.
    # They are never arguments themselves, but removing them here would erase
    # negation, question force, prepositional government and coordination.
    syntactic_indices = tuple(
        index
        for index in range(start, end)
        if cards[index].surface[:1].isalnum() and index not in address_indices
        and cards[index].semantic_role not in {"emotional_modifier", "pragmatic_marker"}
    )
    content_indices = tuple(
        index
        for index in syntactic_indices
        if cards[index].semantic_role != "function_word"
    )
    predicate_index = next(
        (
            index
            for index in content_indices
            if cards[index].part_of_speech == "verb" and "INFN" not in cards[index].grammemes
        ),
        None,
    )
    if predicate_index is None:
        predicate_index = next(
            (index for index in content_indices if cards[index].part_of_speech == "predicate"),
            None,
        )
    imperative = bool(predicate_index is not None and "impr" in cards[predicate_index].grammemes)
    question_target = next(
        (
            index
            for index in syntactic_indices
            if role(cards[index].surface) == "question" or role(cards[index].lemma) == "question"
        ),
        None,
    )
    subject_index = _subject_index(cards, content_indices, predicate_index)
    if imperative and subject_index is not None and subject_index > predicate_index:
        # A post-verbal nominal in an imperative is normally its object
        # (``открой дверь``), never a second addressee alongside the implicit
        # imperative subject.
        subject_index = None
    complements = _complements(cards, syntactic_indices, content_indices)
    coordinations = _coordinations(cards, syntactic_indices, content_indices)
    complement_objects = {
        index for complement in complements for index in complement.object_indices
    }
    predicate_complement: tuple[int, ...] = ()
    implicit_predicate: str | None = None
    if predicate_index is None and subject_index is not None:
        predicate_complement = tuple(
            index
            for index in content_indices
            if index > subject_index
            and index not in complement_objects
            and cards[index].part_of_speech in {"noun", "adjective", "participle"}
        )
        if predicate_complement:
            implicit_predicate = "быть"
        elif question_target is not None and any(
            cards[index].lemma == "такой" for index in syntactic_indices
        ):
            # ``что такое небо`` has an overt subject and an omitted copula;
            # ``такое`` is a definitional scaffold, not a lexical predicate.
            implicit_predicate = "быть"

    direct_object: tuple[int, ...] = ()
    indirect_object: tuple[int, ...] = ()
    action_complements: tuple[int, ...] = ()
    if predicate_index is not None:
        action_complements = tuple(
            index
            for index in content_indices
            if index > predicate_index
            and cards[index].part_of_speech == "verb"
            and "INFN" in cards[index].grammemes
        )
        after = tuple(
            index
            for index in content_indices
            if index > predicate_index
            and index != subject_index
            and index not in complement_objects
            and cards[index].part_of_speech in {"noun", "pronoun"}
        )
        direct = next(
            (index for index in after if _feature(cards[index], "case") == "accusative"),
            None,
        )
        indirect = next(
            (index for index in after if _feature(cards[index], "case") == "dative"),
            None,
        )
        if direct is None:
            direct = next(
                (index for index in after if index != indirect),
                None,
            )
        if direct is None:
            profile = predicate_valency(cards[predicate_index].lemma)
            requires_object = bool(
                profile
                and any(
                    slot.role == "direct_object" and slot.required_in_isolated_fact
                    for slot in profile.slots
                )
            )
            if requires_object:
                direct = next(
                    (
                        index
                        for index in reversed(content_indices)
                        if index < predicate_index
                        and index != subject_index
                        and index not in complement_objects
                        and cards[index].part_of_speech in {"noun", "pronoun"}
                        and _can_be_accusative(cards[index])
                    ),
                    None,
                )
        if direct is None:
            # Russian freely places an object before its predicate: "я тебя
            # спрашиваю", "книгу читаю". Case, not word order,
            # distinguishes it from the subject.
            direct = next(
                (
                    index
                    for index in reversed(content_indices)
                    if index < predicate_index
                    and index != subject_index
                    and index not in complement_objects
                    and cards[index].part_of_speech in {"noun", "pronoun"}
                    and _feature(cards[index], "case") == "accusative"
                ),
                None,
            )
        if indirect is None:
            indirect = next(
                (
                    index
                    for index in content_indices
                    if index < predicate_index
                    and index != subject_index
                    and index not in complement_objects
                    and cards[index].part_of_speech in {"noun", "pronoun"}
                    and _feature(cards[index], "case") == "dative"
                ),
                None,
            )
        direct_object = (direct,) if direct is not None else ()
        indirect_object = (indirect,) if indirect is not None else ()
        direct_object = _expand_coordinated_role(direct_object, coordinations)
        indirect_object = _expand_coordinated_role(indirect_object, coordinations)

    negated = bool(
        predicate_index is not None
        and any(
            cards[index].lemma == "не"
            for index in range(max(start, predicate_index - 2), predicate_index)
        )
    )
    modal_lemmas = tuple(
        dict.fromkeys(
            cards[index].lemma for index in content_indices if cards[index].lemma in _MODAL_LEMMAS
        )
    )
    implicit_subject = (
        "addressee"
        if imperative
        or (
            predicate_index is not None
            and subject_index is None
            and _feature(cards[predicate_index], "person") == "second"
        )
        else "speaker"
        if (
            predicate_index is not None
            and subject_index is None
            and _feature(cards[predicate_index], "person") == "first"
        )
        else "previous_clause_subject"
        if predicate_index is not None and subject_index is None and clause_index > 0
        else None
    )
    elliptical = predicate_index is None and implicit_predicate is None
    question_word_has_main_scope = bool(
        question_target is not None
        and (
            predicate_index is None
            or question_target < predicate_index
        )
    )
    clause_kind = (
        "question"
        if utterance_is_question or question_word_has_main_scope
        else "command"
        if imperative
        else "nominal_statement"
        if implicit_predicate is not None
        else "statement"
        if predicate_index is not None
        else "fragment"
    )
    confidence = 0.24
    confidence += 0.28 if predicate_index is not None or implicit_predicate else 0.0
    confidence += 0.18 if subject_index is not None or implicit_subject else 0.0
    confidence += 0.12 if direct_object or indirect_object or complements else 0.0
    confidence += 0.10 if clause_kind in {"question", "command", "nominal_statement"} else 0.0
    if elliptical:
        confidence -= 0.08
    return ClauseFrame(
        clause_index=clause_index,
        token_start=start,
        token_end=end,
        clause_kind=clause_kind,
        subject_indices=(subject_index,) if subject_index is not None else (),
        predicate_index=predicate_index,
        action_complement_indices=action_complements,
        predicate_complement_indices=predicate_complement,
        direct_object_indices=direct_object,
        indirect_object_indices=indirect_object,
        complements=complements,
        coordinations=coordinations,
        vocative_indices=tuple(index for index in range(start, end) if index in address_indices),
        question_target_index=question_target,
        negated=negated,
        modal_lemmas=modal_lemmas,
        implicit_subject=implicit_subject,
        implicit_predicate=implicit_predicate,
        elliptical=elliptical,
        confidence=max(0.0, min(1.0, confidence)),
    )


def _subject_index(
    cards: tuple[TokenInterpretation, ...],
    content_indices: tuple[int, ...],
    predicate_index: int | None,
) -> int | None:
    before = tuple(
        index for index in content_indices if predicate_index is None or index < predicate_index
    )
    candidates = tuple(
        index
        for index in before
        if (
            cards[index].part_of_speech in {"noun", "pronoun"}
            or cards[index].entity_kind in {"person", "nickname"}
        )
        and _feature(cards[index], "case") in {None, "nominative"}
    )
    if not candidates and predicate_index is not None:
        # Russian questions and colloquial speech freely place a nominative
        # subject after the predicate: "считаешь ты...?".  Do not confuse a
        # later accusative/reflexive object with that inverted subject.
        candidates = tuple(
            index
            for index in content_indices
            if index > predicate_index
            and (
                cards[index].part_of_speech in {"noun", "pronoun"}
                or cards[index].entity_kind in {"person", "nickname"}
            )
            and _feature(cards[index], "case") in {None, "nominative"}
        )
    if not candidates:
        return None
    return candidates[0] if predicate_index is None else candidates[-1]


def _complements(
    cards: tuple[TokenInterpretation, ...],
    syntactic_indices: tuple[int, ...],
    content_indices: tuple[int, ...],
) -> tuple[ClauseComplement, ...]:
    complements: list[ClauseComplement] = []
    content_set = set(content_indices)
    for index in syntactic_indices:
        if cards[index].part_of_speech != "preposition":
            continue
        target = next(
            (
                following
                for following in range(index + 1, min(len(cards), index + 5))
                if following in content_set
                and cards[following].part_of_speech in {"noun", "pronoun", "number"}
            ),
            None,
        )
        if target is None:
            continue
        complements.append(
            ClauseComplement(
                _complement_role(cards[index].lemma),
                index,
                (target,),
            )
        )
    return tuple(complements)


def _coordinations(
    cards: tuple[TokenInterpretation, ...],
    syntactic_indices: tuple[int, ...],
    content_indices: tuple[int, ...],
) -> tuple[ClauseCoordination, ...]:
    """Resolve conjunctions as operators instead of semantic graph nodes."""

    content_set = set(content_indices)
    result: list[ClauseCoordination] = []
    for index in syntactic_indices:
        relation = _COORDINATION_RELATIONS.get(cards[index].lemma)
        if relation is None:
            continue
        left = next(
            (
                candidate
                for candidate in reversed(range(0, index))
                if candidate in content_set
                and cards[candidate].part_of_speech in _COORDINATABLE_PARTS
            ),
            None,
        )
        right = next(
            (
                candidate
                for candidate in range(index + 1, len(cards))
                if candidate in content_set
                and cards[candidate].part_of_speech in _COORDINATABLE_PARTS
            ),
            None,
        )
        if left is None or right is None:
            continue
        result.append(ClauseCoordination(relation, index, (left, right)))
    return tuple(result)


def _expand_coordinated_role(
    indices: tuple[int, ...],
    coordinations: tuple[ClauseCoordination, ...],
) -> tuple[int, ...]:
    expanded = list(indices)
    changed = True
    while changed:
        changed = False
        known = set(expanded)
        for coordination in coordinations:
            if not (known & set(coordination.member_indices)):
                continue
            for member in coordination.member_indices:
                if member not in known:
                    expanded.append(member)
                    known.add(member)
                    changed = True
    return tuple(sorted(expanded))


def _complement_role(preposition: str) -> str:
    if preposition in _TOPIC_PREPOSITIONS:
        return "topic"
    if preposition in _SOURCE_PREPOSITIONS:
        return "source_or_companion" if preposition in _INSTRUMENT_PREPOSITIONS else "source"
    if preposition in _LOCATION_PREPOSITIONS:
        return "location_or_destination"
    if preposition in _DESTINATION_PREPOSITIONS:
        return "destination_or_recipient"
    return "prepositional_complement"


def _roles_for_frame(frame: ClauseFrame) -> dict[int, str]:
    roles: dict[int, str] = {}
    for index in frame.vocative_indices:
        roles[index] = "vocative"
    for index in frame.subject_indices:
        roles[index] = "subject"
    if frame.predicate_index is not None:
        roles[frame.predicate_index] = "predicate"
    for index in frame.action_complement_indices:
        roles[index] = "action_complement"
    for index in frame.predicate_complement_indices:
        roles[index] = "predicate_complement"
    for index in frame.direct_object_indices:
        roles[index] = "direct_object"
    for index in frame.indirect_object_indices:
        roles[index] = "indirect_object"
    for complement in frame.complements:
        roles[complement.preposition_index] = f"complement_link:{complement.role}"
        for index in complement.object_indices:
            roles[index] = f"complement:{complement.role}"
    if frame.question_target_index is not None:
        roles[frame.question_target_index] = "question_target"
    return roles


def _dependency_arcs(
    cards: tuple[TokenInterpretation, ...],
    frames: tuple[ClauseFrame, ...],
) -> tuple[DependencyArc, ...]:
    """Project clause roles into a compact dependency graph.

    This intentionally uses only justified overt-token edges.  Implicit
    subjects and copulas stay on ``ClauseFrame`` instead of being represented
    by fake token indexes.
    """

    arcs: list[DependencyArc] = []
    for frame in frames:
        head = frame.predicate_index
        if head is None and frame.subject_indices:
            head = frame.subject_indices[0]
        if head is None:
            continue

        def connect(
            indices: tuple[int, ...],
            relation: str,
            *,
            clause_index: int = frame.clause_index,
            dependency_head: int = head,
        ) -> None:
            arcs.extend(
                DependencyArc(clause_index, dependency_head, index, relation)
                for index in indices
                if index != dependency_head
            )

        connect(frame.subject_indices, "nsubj")
        connect(frame.action_complement_indices, "xcomp")
        for object_index in frame.direct_object_indices:
            object_head = next(
                (
                    index
                    for index in reversed(frame.action_complement_indices)
                    if index < object_index
                ),
                head,
            )
            if object_index != object_head:
                arcs.append(
                    DependencyArc(
                        frame.clause_index,
                        object_head,
                        object_index,
                        "obj",
                    )
                )
        connect(frame.indirect_object_indices, "iobj")
        connect(
            frame.predicate_complement_indices,
            "copular_complement" if frame.predicate_index is None else "xcomp",
        )
        for complement in frame.complements:
            for object_index in complement.object_indices:
                if object_index != head:
                    arcs.append(
                        DependencyArc(
                            frame.clause_index,
                            head,
                            object_index,
                            f"obl:{complement.role}",
                        )
                    )
                arcs.append(
                    DependencyArc(
                        frame.clause_index,
                        object_index,
                        complement.preposition_index,
                        "case",
                    )
                )
        if frame.question_target_index is not None and frame.question_target_index != head:
            arcs.append(
                DependencyArc(
                    frame.clause_index,
                    head,
                    frame.question_target_index,
                    "interrogative",
                )
            )
        if frame.negated and frame.predicate_index is not None:
            negation_index = next(
                (
                    index
                    for index in range(
                        max(frame.token_start, frame.predicate_index - 2),
                        frame.predicate_index,
                    )
                    if cards[index].lemma == "не"
                ),
                None,
            )
            if negation_index is not None:
                arcs.append(
                    DependencyArc(
                        frame.clause_index,
                        head,
                        negation_index,
                        "advmod:neg",
                    )
                )
        for coordination in frame.coordinations:
            coordination_head = coordination.member_indices[0]
            for member in coordination.member_indices[1:]:
                arcs.append(
                    DependencyArc(
                        frame.clause_index,
                        coordination_head,
                        member,
                        f"conj:{coordination.relation}",
                    )
                )
            arcs.append(
                DependencyArc(
                    frame.clause_index,
                    coordination_head,
                    coordination.conjunction_index,
                    "cc",
                )
            )

        argument_indices = tuple(
            dict.fromkeys(
                (
                    *frame.subject_indices,
                    *frame.action_complement_indices,
                    *frame.direct_object_indices,
                    *frame.indirect_object_indices,
                    *frame.predicate_complement_indices,
                    *(index for item in frame.complements for index in item.object_indices),
                )
            )
        )
        for argument_index in argument_indices:
            for modifier_index in range(max(frame.token_start, argument_index - 2), argument_index):
                if cards[modifier_index].part_of_speech in {"adjective", "participle"}:
                    arcs.append(
                        DependencyArc(
                            frame.clause_index,
                            argument_index,
                            modifier_index,
                            "amod",
                        )
                    )
    return tuple(dict.fromkeys(arcs))


def _clause_ranges(
    tokens: tuple[str, ...],
    cards: tuple[TokenInterpretation, ...],
) -> tuple[tuple[int, int], ...]:
    sentence_ranges: list[tuple[int, int]] = []
    start = 0
    for index, token in enumerate(tokens):
        if token not in _CLAUSE_END:
            continue
        if any(item[:1].isalnum() for item in tokens[start:index]):
            sentence_ranges.append((start, index))
        start = index + 1
    if any(item[:1].isalnum() for item in tokens[start:]):
        sentence_ranges.append((start, len(tokens)))

    ranges: list[tuple[int, int]] = []
    for sentence_start, sentence_end in sentence_ranges:
        boundaries: list[tuple[int, int]] = []
        for index in range(sentence_start + 1, sentence_end - 1):
            if (
                cards[index].lemma in {"и", "а", "но", "зато"}
                and _has_clause_core(cards, sentence_start, index)
                and _has_clause_core(cards, index + 1, sentence_end)
            ):
                boundaries.append((index, index + 1))
                continue
            if tokens[index] != ",":
                continue
            marker_index = next(
                (
                    following
                    for following in range(index + 1, min(sentence_end, index + 3))
                    if cards[following].surface[:1].isalnum()
                ),
                None,
            )
            if (
                marker_index is not None
                and cards[marker_index].lemma in _SUBORDINATE_MARKERS
                and _has_clause_core(cards, sentence_start, index)
                and _has_clause_core(cards, marker_index, sentence_end)
            ):
                boundaries.append((index, marker_index))
        clause_start = sentence_start
        for split_end, next_start in sorted(dict.fromkeys(boundaries)):
            if split_end < clause_start or not _has_clause_core(cards, clause_start, split_end):
                continue
            ranges.append((clause_start, split_end))
            clause_start = next_start
        if any(item[:1].isalnum() for item in tokens[clause_start:sentence_end]):
            ranges.append((clause_start, sentence_end))
    return tuple(ranges)


def _has_clause_core(
    cards: tuple[TokenInterpretation, ...],
    start: int,
    end: int,
) -> bool:
    if _has_finite_predicate(cards, start, end):
        return True
    lexical = tuple(card for card in cards[start:end] if card.surface[:1].isalnum())
    has_nominal = any(card.part_of_speech in {"noun", "pronoun"} for card in lexical)
    has_question = any(
        role(card.surface) == "question" or role(card.lemma) == "question"
        for card in lexical
    )
    definitional = any(card.lemma == "такой" for card in lexical)
    return has_nominal and (has_question or definitional)


def _has_finite_predicate(
    cards: tuple[TokenInterpretation, ...],
    start: int,
    end: int,
) -> bool:
    return any(
        card.part_of_speech == "verb" and "INFN" not in card.grammemes for card in cards[start:end]
    )


def _clause_links(
    cards: tuple[TokenInterpretation, ...],
    frames: tuple[ClauseFrame, ...],
) -> tuple[ClauseLink, ...]:
    links: list[ClauseLink] = []
    for child_index, child in enumerate(frames[1:], start=1):
        first_lexical = next(
            (
                index
                for index in range(child.token_start, child.token_end)
                if cards[index].surface[:1].isalnum()
            ),
            None,
        )
        if first_lexical is None:
            continue
        subordinate_relation = _SUBORDINATE_MARKERS.get(cards[first_lexical].lemma)
        if subordinate_relation is not None:
            links.append(
                ClauseLink(child_index - 1, child_index, subordinate_relation, first_lexical)
            )
            continue
        conjunction_index = child.token_start - 1
        if conjunction_index >= 0:
            coordination = _COORDINATION_RELATIONS.get(cards[conjunction_index].lemma)
            if coordination is not None:
                parent_index = child_index - 1
                parent_link = next(
                    (
                        item
                        for item in reversed(links)
                        if item.child_clause_index == parent_index
                        and item.relation.startswith("ccomp")
                    ),
                    None,
                )
                if parent_link is not None:
                    parent_index = parent_link.parent_clause_index
                links.append(
                    ClauseLink(
                        parent_index,
                        child_index,
                        f"coord:{coordination}",
                        conjunction_index,
                    )
                )
    return tuple(links)


def _entities_by_index(
    tokens: tuple[str, ...], mentions: tuple[EntityMention, ...]
) -> dict[int, EntityMention]:
    result: dict[int, EntityMention] = {}
    for index, token in enumerate(tokens):
        normalized = normalize_word(token)
        matches = tuple(
            mention
            for mention in mentions
            if normalized == normalize_word(mention.canonical)
            or normalized in {normalize_word(item) for item in mention.surface.split()}
        )
        if matches:
            result[index] = max(matches, key=lambda item: item.confidence)
    return result


def _pragmatics_by_index(tokens, cues) -> dict[int, tuple[str, ...]]:
    result: dict[int, list[str]] = {}
    for cue in cues:
        cue_terms = {normalize_word(token) for token in cue.surface.split()}
        for index, token in enumerate(tokens):
            if normalize_word(token) in cue_terms:
                result.setdefault(index, []).append(cue.function)
    return {index: tuple(dict.fromkeys(functions)) for index, functions in result.items()}


def _quoted_indices(tokens: tuple[str, ...]) -> frozenset[int]:
    quoted: set[int] = set()
    active = False
    for index, token in enumerate(tokens):
        if token in _QUOTE_TOKENS:
            quoted.add(index)
            active = not active
        elif active:
            quoted.add(index)
    return frozenset(quoted)


def _lexical_index(analyses: tuple[WordAnalysis, ...]) -> dict[str, WordAnalysis]:
    result: dict[str, WordAnalysis] = {}
    for analysis in analyses:
        for key in (
            normalize_word(analysis.surface_form),
            normalize_word(analysis.normalized_form),
            normalize_word(analysis.lemma or ""),
        ):
            if key:
                result.setdefault(key, analysis)
    return result


def _selected_meaning(analysis: WordAnalysis | None) -> str | None:
    if analysis is None:
        return None
    return analysis.local_meaning or analysis.selected_meaning or analysis.personal_interpretation


def _normalized_part(analysis: WordAnalysis | None) -> str | None:
    if analysis is None or analysis.part_of_speech is None:
        return None
    part = analysis.part_of_speech.casefold()
    aliases = {"noun": "noun", "npro": "pronoun", "verb": "verb", "infn": "verb"}
    return aliases.get(part, part)


def _token_confidence(
    reading: MorphologicalReading | None,
    lexical: WordAnalysis | None,
    entity: EntityMention | None,
) -> float:
    if entity is not None:
        return entity.confidence
    morphology = reading.score if reading is not None else 0.0
    lexical_confidence = lexical.confidence.form if lexical is not None else 0.0
    if reading is not None and lexical is not None:
        return min(1.0, morphology * 0.55 + lexical_confidence * 0.45)
    return max(morphology, lexical_confidence)


def _feature(card: TokenInterpretation, name: str) -> str | None:
    return dict(card.grammatical_features).get(name)


def _can_be_accusative(card: TokenInterpretation) -> bool:
    if _feature(card, "case") == "accusative":
        return True
    return any(
        reading.lemma == card.lemma
        and reading.part_of_speech == card.part_of_speech
        and reading.case == "accusative"
        for reading in DEFAULT_MORPHOLOGY.parse_word(card.corrected_form or card.surface)
    )
