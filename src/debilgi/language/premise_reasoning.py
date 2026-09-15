"""Small, explicit inference chains from lexical senses and the current roles.

Definitions supply premises, never complete answers. Ambiguous senses stay
conditional; no inferred preference or hypothetical fact is written as a lesson.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from debilgi.language.communication import MessagePlan, ResponseComposer
from debilgi.language.lexicon import WordAnalysis
from debilgi.language.morphology import DEFAULT_MORPHOLOGY, MorphologicalReading
from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    ProductiveSurface,
    SemanticDependency,
)
from debilgi.language.structured_utterance import StructuredUtterance, analyze_structured_utterance

_TOOLS = frozenset({"инструмент", "прибор", "устройство", "механизм"})
_BODY_WHOLES = frozenset({"тело", "лицо", "морда", "организм"})
_BIOLOGICAL_FUNCTIONS = frozenset({
    "пищеварение", "дыхание", "обоняние", "зрение", "слух", "осязание", "кровообращение",
})


@dataclass(frozen=True, slots=True)
class LexicalPremise:
    category: ConceptArgument
    action: str | None
    bodily: bool
    evidence: str


@dataclass(frozen=True, slots=True)
class PremiseInference:
    kind: str
    target: str
    conclusion: ClauseThought
    reason: ClauseThought
    assumption: ClauseThought | None
    evidence: tuple[str, ...]


def _lexeme(reading: MorphologicalReading, surface: str) -> ConceptLexeme:
    return ConceptLexeme(
        surface if reading.part_of_speech == "participle" else reading.lemma,
        surface, reading.part_of_speech, tuple(reading.features().items()),
    )


def lexical_premise(definition: str) -> LexicalPremise | None:
    """Read a leading class phrase, including active functional modifiers.

    A mention later in a definition is not a type: 'idea about an instrument'
    must not make the idea into a physical tool.
    """
    sentence = DEFAULT_MORPHOLOGY.analyze_sentence(definition.replace("\u0301", ""))
    pairs = tuple(zip(sentence.tokens, sentence.readings, strict=True))
    if any(token.casefold() in {"не", "никогда", "возможно", "якобы"} for token, _ in pairs):
        return None
    head_index = next((i for i, (_, r) in enumerate(pairs)
                       if r is not None and r.part_of_speech == "noun"), None)
    if head_index is None or head_index > 3:
        return None
    modifiers = pairs[:head_index]
    if any(r is None or r.part_of_speech not in {"adjective", "participle"}
           for _, r in modifiers):
        return None
    surface, head = pairs[head_index]
    post = ()
    if head.lemma == "часть" and head_index + 1 < len(pairs):
        parent_surface, parent = pairs[head_index + 1]
        parent = next((r for r in DEFAULT_MORPHOLOGY.parse_word(parent_surface)
                       if r.part_of_speech == "noun" and r.features().get("case") == "genitive"),
                      None)
        if parent is not None and parent.features().get("case") == "genitive":
            post = (_lexeme(parent, parent_surface),)
    # 'Organ' can also name an institution. Only a biological dependent or
    # modifier licenses the biological premise.
    biological_organ = head.lemma == "орган" and (
        any(r.lemma in {"анатомический", "внутренний", "биологический"}
            for _, r in modifiers)
        or (head_index + 1 < len(pairs) and any(
            r.lemma in _BIOLOGICAL_FUNCTIONS and r.features().get("case") == "genitive"
            for r in DEFAULT_MORPHOLOGY.parse_word(pairs[head_index + 1][0])
        ))
    )
    bodily = biological_organ or head.lemma == "конечность" or (
        head.lemma == "часть" and post and post[0].lemma in _BODY_WHOLES
    )
    actions = tuple(dict.fromkeys(
        r.lemma for token, _ in modifiers for r in DEFAULT_MORPHOLOGY.parse_word(token)
        if r.part_of_speech == "participle" and "actv" in r.grammemes
    ))
    action = actions[0] if head.lemma in _TOOLS and len(actions) == 1 else None
    if not bodily and action is None:
        return None
    return LexicalPremise(
        ConceptArgument(_lexeme(head, surface),
                        modifiers=tuple(_lexeme(r, token) for token, r in modifiers),
                        postmodifiers=post),
        action, bool(bodily), definition,
    )


def _thought(
    predicate: str, dependencies: tuple[SemanticDependency, ...], evidence: tuple[str, ...],
    *, negated: bool = False, modal: bool = False,
) -> ClauseThought:
    return ClauseThought(
        ConceptLexeme(predicate, predicate, "verb"), tuple(dependencies),
        negated=negated, confidence=0.88, evidence=evidence,
        predicate_class="modal" if modal else "lexical",
    )


def _argument(
    lemma: str, part: str = "noun", features: tuple[tuple[str, str], ...] = (),
) -> ConceptArgument:
    return ConceptArgument(ConceptLexeme(lemma, lemma, part, features))


def infer_from_premises(
    structure: StructuredUtterance, analyses: tuple[WordAnalysis, ...],
) -> PremiseInference | None:
    if len(structure.clauses) != 1:
        return None
    clause = structure.clauses[0]
    if clause.clause_kind != "question" or clause.predicate_index is None:
        return None
    if any(t.lemma in {"если", "представить", "допустить", "бы"} for t in structure.tokens):
        return None
    predicate = structure.tokens[clause.predicate_index]
    target_index = None
    kind = None
    if predicate.lemma in {"думать", "полагать"} and (
        clause.question_target_index is not None
        and structure.tokens[clause.question_target_index].lemma == "что"
    ) and (
        clause.implicit_subject == "addressee"
        or any(structure.tokens[i].lemma in {"ты", "вы"} for i in clause.subject_indices)
    ) and not clause.negated:
        topics = tuple(i for comp in clause.complements
                       if structure.tokens[comp.preposition_index].lemma in {"о", "об", "про"}
                       for i in comp.object_indices if structure.tokens[i].part_of_speech == "noun")
        if len(topics) == 1:
            target_index, kind = topics[0], "functional_appraisal"
    elif len(clause.subject_indices) == 1:
        owners = tuple(comp for comp in clause.complements
                       if structure.tokens[comp.preposition_index].lemma == "у"
                       and any(structure.tokens[i].lemma in {"ты", "вы"}
                               for i in comp.object_indices))
        if len(owners) == 1 and len(clause.complements) == 1:
            target_index, kind = clause.subject_indices[0], "bodily_presupposition"
    if target_index is None:
        return None
    target = structure.tokens[target_index]
    if target.part_of_speech != "noun":
        return None
    # Modified/mediated entities require their own model: a virtual organ or an
    # organ on a screen is not the bot's biological body.
    if any(t.part_of_speech in {"adjective", "participle"} for t in structure.tokens):
        return None
    word = next((a for a in analyses if a.lemma == target.lemma), None)
    if word is None:
        return None
    chosen = word.personal_interpretation or word.local_meaning or word.selected_meaning
    meanings = (chosen,) if chosen else word.dictionary_meanings
    premises = tuple(p for meaning in meanings if (p := lexical_premise(meaning)) is not None)
    premise = next((p for p in premises if p.action) if kind == "functional_appraisal"
                   else (p for p in premises if p.bodily), None)
    if premise is None:
        return None
    evidence = (
        f"lexical_premise:{target.lemma}:{premise.evidence}",
        "inference:function_enables_action" if kind == "functional_appraisal"
        else "agent_ontology:no_biological_body",
    )
    subject = SemanticDependency("subject", _argument("я", "pronoun"))
    obj = _argument(target.lemma, features=target.grammatical_features)
    if kind == "functional_appraisal":
        conclusion = _thought("считать", (
            subject, SemanticDependency("direct_object", obj),
            SemanticDependency("predicate_complement", _argument(
                "полезный", "adjective", target.grammatical_features,
            )),
        ), evidence)
        reason = _thought("позволять", (
            SemanticDependency("subject", obj),
            SemanticDependency("action_complement", _argument(premise.action, "verb")),
        ), evidence, modal=True)
    else:
        conclusion = _thought("иметь", (
            subject, SemanticDependency("direct_object", replace(obj, grammatical_case="genitive")),
        ), evidence, negated=True)
        reason = _thought("иметь", (
            subject, SemanticDependency("direct_object", replace(
                _argument("тело"), grammatical_case="genitive",
            )),
        ), evidence, negated=True)
    assumption = None
    if not chosen and len(meanings) > 1:
        assumption = _thought("идти", (
            SemanticDependency("subject", _argument("речь")),
            SemanticDependency("topic", replace(
                premise.category, preposition="о", grammatical_case="prepositional",
            )),
        ), evidence)
    return PremiseInference(kind, target.lemma, conclusion, reason, assumption, evidence)


def realize_premise_inference(
    inference: PremiseInference, *, composer: ResponseComposer | None = None,
) -> ProductiveSurface | None:
    composer = composer or ResponseComposer(analyze_structured_utterance(""))
    surfaces = composer.compose(MessagePlan(
        composer.goal, inference.conclusion, inference.reason, inference.assumption,
    ), limit=1)
    return replace(surfaces[0], source="premise_inference") if surfaces else None
