"""General relation questions grounded in morphology, lexical types and subject facts.

The first supported relation is possession/existential availability.  The
implementation intentionally reasons over a frame (holder, relation, object)
instead of matching complete authored questions or individual object names.
"""

from __future__ import annotations

from dataclasses import dataclass

from debilgi.language.construction_nlg import (
    ResponseContentPlan,
    SemanticClause,
    SemanticTerm,
    realize_content_plan,
)
from debilgi.language.content import content_lemmas
from debilgi.language.lexicon import WordAnalysis
from debilgi.language.morphology import DEFAULT_MORPHOLOGY
from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    SemanticDependency,
)
from debilgi.language.structured_utterance import StructuredUtterance

_POSSESSION_PREDICATES = frozenset({"быть", "есть", "иметь", "иметься", "обладать"})
_PHYSICAL_TYPE_TERMS = frozenset(
    {
        "вещь",
        "предмет",
        "изделие",
        "материал",
        "материальный",
        "тело",
        "одежда",
        "обувь",
        "ткань",
        "инструмент",
        "устройство",
        "механизм",
        "сосуд",
        "помещение",
        "продукт",
        "пища",
    }
)
_COGNITIVE_TYPE_TERMS = frozenset(
    {
        "память",
        "мышление",
        "мысль",
        "знание",
        "информация",
        "представление",
        "суждение",
        "мнение",
        "намерение",
        "цель",
    }
)
_IDENTITY_TYPE_TERMS = frozenset(
    {"имя", "название", "псевдоним", "ник", "идентификатор", "идентичность"}
)
_DIGITAL_TYPE_TERMS = frozenset(
    {"файл", "данные", "программа", "цифровой", "электронный", "код", "запись"}
)
_ABSTRACT_TYPE_TERMS = frozenset(
    {"действие", "процесс", "состояние", "понятие", "свойство", "явление", "отношение"}
)
_POSSESSIBLE_CLASSES = frozenset({"physical", "cognitive", "identity", "digital"})


@dataclass(frozen=True, slots=True)
class RelationQuestionFrame:
    relation: str
    holder_lemma: str
    holder_surface: str
    holder_kind: str
    object_surface: str
    object_lemmas: tuple[str, ...]
    object_number: str | None
    query_negated: bool
    complete: bool
    confidence: float

    @property
    def study_terms(self) -> tuple[str, ...]:
        return self.object_lemmas


@dataclass(frozen=True, slots=True)
class RelationReasoningResult:
    content_plan: ResponseContentPlan | None
    confidence: float
    relation: str
    object_lemma: str | None
    object_class: str | None
    needs_study: bool
    evidence_terms: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        """Compatibility view; wording belongs to the construction realizer."""

        surfaces = realize_content_plan(self.content_plan) if self.content_plan else ()
        return surfaces[0].text if surfaces else ""

    @property
    def productive_thought(self) -> ClauseThought | None:
        """Translate relation truth into roles without choosing a sentence shell."""

        if self.content_plan is None or len(self.content_plan.clauses) != 1:
            return None
        clause = self.content_plan.clauses[0]
        if clause.relation != "possession" or clause.truth is None:
            return None
        item_case = "genitive" if clause.truth is False else "nominative"
        holder = ConceptArgument(
            ConceptLexeme(
                clause.subject.lemma,
                clause.subject.surface,
                clause.subject.part_of_speech,
                (("number", clause.subject.number),) if clause.subject.number else (),
                clause.subject.entity_kind,
            ),
            preposition="у",
            grammatical_case="genitive",
        )
        item = ConceptArgument(
            ConceptLexeme(
                clause.object.lemma,
                clause.object.surface,
                clause.object.part_of_speech,
                (("number", clause.object.number),) if clause.object.number else (),
                clause.object.entity_kind,
            ),
            grammatical_case=item_case,
        )
        return ClauseThought(
            predicate=ConceptLexeme(
                "нет" if clause.truth is False else "есть",
                "нет" if clause.truth is False else "есть",
                "existential",
            ),
            dependencies=(
                SemanticDependency("location", holder),
                SemanticDependency("subject", item),
            ),
            response_polarity=self.content_plan.answer_polarity,
            focus_role="subject",
            confidence=self.content_plan.confidence,
            evidence=self.content_plan.evidence,
            predicate_class="existential",
        )


def analyze_relation_question(
    structure: StructuredUtterance,
) -> RelationQuestionFrame | None:
    """Extract a possession frame from direct, inverted or embedded wording."""

    words = tuple(token for token in structure.tokens if token.surface[:1].isalnum())
    if not words:
        return None
    question_like = bool(
        structure.text.rstrip().endswith("?")
        or any(frame.clause_kind == "question" for frame in structure.clauses)
        or any(token.lemma in {"ли", "разве", "неужели"} for token in words)
    )
    if not question_like:
        return None

    predicate = next(
        (
            token
            for token in words
            if token.lemma in _POSSESSION_PREDICATES
            or token.normalized in {"есть", "имеется", "имеет", "имеешь", "имею", "нет"}
        ),
        None,
    )
    if predicate is None:
        return None
    query_negated = predicate.normalized == "нет" or any(
        token.lemma == "не" and token.index < predicate.index for token in words
    )

    holder = None
    possession_preposition = next(
        (token for token in words if token.lemma == "у"),
        None,
    )
    if possession_preposition is not None:
        holder = next(
            (
                token
                for token in words
                if token.index > possession_preposition.index
                and token.index <= possession_preposition.index + 2
                and token.part_of_speech in {"noun", "pronoun"}
            ),
            None,
        )
    if holder is None and predicate.lemma in {"иметь", "обладать"}:
        holder = next(
            (
                token
                for token in reversed(words)
                if token.index < predicate.index
                and token.part_of_speech in {"noun", "pronoun"}
            ),
            None,
        )
    if holder is None:
        return None

    excluded = {holder.index, predicate.index}
    if possession_preposition is not None:
        excluded.add(possession_preposition.index)
    after_predicate = tuple(
        token
        for token in words
        if token.index > predicate.index
        and token.index not in excluded
        and token.part_of_speech == "noun"
    )
    before_holder = tuple(
        token
        for token in words
        if token.index < (possession_preposition.index if possession_preposition else holder.index)
        and token.index not in excluded
        and token.part_of_speech == "noun"
    )
    object_token = (after_predicate or before_holder[-1:])[-1] if (
        after_predicate or before_holder
    ) else None
    if object_token is None:
        return None

    readings = tuple(
        item
        for item in DEFAULT_MORPHOLOGY.parse_word(object_token.surface)
        if item.part_of_speech == "noun"
        and item.case in {None, "nominative", "accusative"}
    )
    object_lemmas = tuple(
        dict.fromkeys((*(item.lemma for item in readings), object_token.lemma))
    )
    object_number = next((item.number for item in readings if item.number), None)
    holder_kind = (
        "self"
        if holder.lemma in {"ты", "вы"}
        else "speaker"
        if holder.lemma in {"я", "мы"}
        else "other"
    )
    confidence = min(
        0.98,
        0.48
        + predicate.confidence * 0.16
        + holder.confidence * 0.14
        + object_token.confidence * 0.14
        + (0.08 if possession_preposition is not None else 0.04),
    )
    return RelationQuestionFrame(
        relation="possession",
        holder_lemma=holder.lemma,
        holder_surface=holder.surface.casefold(),
        holder_kind=holder_kind,
        object_surface=object_token.surface.casefold(),
        object_lemmas=object_lemmas,
        object_number=object_number,
        query_negated=query_negated,
        complete=bool(object_lemmas),
        confidence=confidence,
    )


def reason_about_relation(
    frame: RelationQuestionFrame,
    analyses: tuple[WordAnalysis, ...],
    learned_concepts: tuple[tuple[str, str], ...] = (),
    *,
    study_completed: bool = False,
) -> RelationReasoningResult:
    """Resolve the frame through lexical types and a minimal subject ontology."""

    evidence: list[tuple[float, str, str | None, tuple[str, ...]]] = []
    for lemma in frame.object_lemmas:
        definitions: list[str] = []
        for analysis in analyses:
            if lemma not in {analysis.lemma, analysis.normalized_form}:
                continue
            definitions.extend(
                item
                for item in (
                    analysis.selected_meaning,
                    analysis.local_meaning,
                    analysis.personal_interpretation,
                    *analysis.dictionary_meanings,
                )
                if item
            )
        for title, definition in learned_concepts:
            title_terms = set(content_lemmas(title)) | {title.casefold().replace("ё", "е")}
            if lemma in title_terms and definition:
                definitions.append(definition)
        terms = {
            term
            for definition in definitions[:5]
            for term in content_lemmas(definition)
        }
        object_class, class_score, class_terms = classify_object_terms(terms)
        possessibility = 0.24 if object_class in _POSSESSIBLE_CLASSES else -0.12
        evidence.append(
            (
                class_score + possessibility + min(0.18, len(terms) * 0.008),
                lemma,
                object_class,
                class_terms,
            )
        )
    evidence.sort(key=lambda item: item[0], reverse=True)
    best = evidence[0] if evidence else None
    if best is None or best[2] is None:
        actual = False if frame.holder_kind == "self" and study_completed else None
        return RelationReasoningResult(
            content_plan=(
                _possession_plan(frame, actual, max(0.58, frame.confidence * 0.78), ())
                if actual is not None
                else None
            ),
            confidence=max(0.46, frame.confidence * 0.70),
            relation=frame.relation,
            object_lemma=best[1] if best else None,
            object_class=None,
            needs_study=not study_completed,
        )

    _, object_lemma, object_class, class_terms = best
    if frame.holder_kind != "self":
        confidence = max(0.58, frame.confidence * 0.78)
        return RelationReasoningResult(
            content_plan=_possession_plan(
                frame,
                None,
                confidence,
                class_terms,
                object_lemma=object_lemma,
            ),
            confidence=confidence,
            relation=frame.relation,
            object_lemma=object_lemma,
            object_class=object_class,
            needs_study=False,
            evidence_terms=class_terms,
        )

    if object_class == "physical":
        return RelationReasoningResult(
            content_plan=_possession_plan(
                frame,
                False,
                0.96,
                class_terms,
                object_lemma=object_lemma,
            ),
            confidence=0.96,
            relation=frame.relation,
            object_lemma=object_lemma,
            object_class=object_class,
            needs_study=False,
            evidence_terms=class_terms,
        )
    if object_class in {"cognitive", "identity"}:
        return RelationReasoningResult(
            content_plan=_possession_plan(
                frame,
                True,
                0.91,
                class_terms,
                object_lemma=object_lemma,
            ),
            confidence=0.91,
            relation=frame.relation,
            object_lemma=object_lemma,
            object_class=object_class,
            needs_study=False,
            evidence_terms=class_terms,
        )
    confidence = max(0.58, frame.confidence * 0.78)
    return RelationReasoningResult(
        content_plan=_possession_plan(
            frame,
            False,
            confidence,
            class_terms,
            object_lemma=object_lemma,
        ),
        confidence=confidence,
        relation=frame.relation,
        object_lemma=object_lemma,
        object_class=object_class,
        needs_study=False,
        evidence_terms=class_terms,
    )


def _possession_plan(
    frame: RelationQuestionFrame,
    actual: bool | None,
    confidence: float,
    evidence: tuple[str, ...],
    *,
    object_lemma: str | None = None,
) -> ResponseContentPlan:
    holder = (
        SemanticTerm("я", "я", part_of_speech="pronoun")
        if frame.holder_kind == "self"
        else SemanticTerm("ты", "ты", part_of_speech="pronoun")
        if frame.holder_kind == "speaker"
        else SemanticTerm(frame.holder_lemma, frame.holder_surface)
    )
    selected_object_lemma = object_lemma or frame.object_lemmas[0]
    selected_number = next(
        (
            reading.number
            for reading in DEFAULT_MORPHOLOGY.parse_word(frame.object_surface)
            if reading.part_of_speech == "noun"
            and reading.lemma == selected_object_lemma
            and reading.number is not None
        ),
        frame.object_number,
    )
    item = SemanticTerm(
        selected_object_lemma,
        frame.object_surface,
        number=selected_number,
    )
    answer_polarity = None if actual is None else actual != frame.query_negated
    return ResponseContentPlan(
        response_act="relation_answer",
        answer_polarity=answer_polarity,
        clauses=(SemanticClause("possession", holder, item, actual),),
        confidence=confidence,
        evidence=evidence,
    )


def classify_object_terms(
    terms: set[str],
) -> tuple[str | None, float, tuple[str, ...]]:
    """Classify semantic definition terms for reuse across reasoners."""
    classes = (
        ("physical", terms & _PHYSICAL_TYPE_TERMS),
        ("cognitive", terms & _COGNITIVE_TYPE_TERMS),
        ("identity", terms & _IDENTITY_TYPE_TERMS),
        ("digital", terms & _DIGITAL_TYPE_TERMS),
        ("abstract", terms & _ABSTRACT_TYPE_TERMS),
    )
    ranked = sorted(classes, key=lambda item: len(item[1]), reverse=True)
    best_name, best_terms = ranked[0]
    runner_size = len(ranked[1][1])
    if not best_terms or len(best_terms) == runner_size:
        return None, 0.0, ()
    score = min(0.92, 0.48 + len(best_terms) * 0.16 - runner_size * 0.08)
    return best_name, score, tuple(sorted(best_terms))
