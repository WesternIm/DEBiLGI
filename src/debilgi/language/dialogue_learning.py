"""Learn adjacent human exchanges as acts and grammatical slots, not agent facts.

Closed-class words and social act lexemes can be learned as language. Content
slots retain their morphology and their binding to the *prompt*. An unbound
content slot cannot be filled from a corpus participant's biography.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections import OrderedDict
from contextlib import suppress
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from datetime import datetime, timedelta
from functools import lru_cache

from debilgi.cognition.affect import lexical_valence
from debilgi.events.models import Event, EventType
from debilgi.language import syntax_backend
from debilgi.language.dialogue_acts import analyze_dialogue_act, analyze_turn_form
from debilgi.language.dialogue_state import unbound_anaphors
from debilgi.language.morphology import DEFAULT_MORPHOLOGY
from debilgi.language.pragmatics import analyze_pragmatics
from debilgi.language.productive_nlg import identity_graph
from debilgi.language.structured_utterance import (
    _COORDINATION_RELATIONS,
    analytic_future_events,
    analyze_structured_utterance,
)
from debilgi.social.signals import evidence_counts
from debilgi.storage.database import Database

# Deixis of the agent and of the asker's own matters: never a topic to be curious about (4.38).
_DEIXIS = frozenset({"я", "мы", "ты", "вы", "мой", "наш", "твой", "ваш", "свой"})


def _letter_cases(text: str) -> tuple[str, str, str]:
    """SQLite LIKE ignores case for ASCII only: the lower, capitalised and upper spellings."""
    return (text.lower(), text.lower().capitalize(), text.upper())


def _plan_topics(plan_json: str) -> frozenset:
    """Subject lemmas of the claim of a stored plan (what a curiosity question was about)."""
    plan = json.loads(plan_json)
    claim = plan.get("claim") or {}
    return frozenset((
        *(((d.get("argument") or {}).get("head") or {}).get("lemma") or ""
          for d in claim.get("dependencies") or () if d.get("role") in {"subject", "topic"}),
        *((plan.get("goal") or {}).get("topics") or ())))

_MESSAGE_TYPES = {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED}
_CLOSED = frozenset({"preposition", "conjunction", "particle", "pronoun"})


# Cache identity covers transitive parser code and configuration, including NLP assets.
def _parser_fingerprint():
    import importlib.metadata
    from pathlib import Path

    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    for name in ("pymorphy3", "pymorphy3-dicts-ru"):
        with suppress(importlib.metadata.PackageNotFoundError):
            digest.update(importlib.metadata.version(name).encode())
    return digest.hexdigest()


_ANALYSIS_SOURCE_VERSION = _parser_fingerprint()
ANALYSIS_VERSION = _ANALYSIS_SOURCE_VERSION + ":" + syntax_backend._analysis_identity
_ANALYSIS_PAYLOADS = OrderedDict()
_LOADED_ANALYSIS_VERSION = None
_PREPARED_EXCHANGES = OrderedDict()


def configure_analysis_cache(backend_key):
    """Parser configuration is part of projection identity, as well as parser code."""
    global ANALYSIS_VERSION, _LOADED_ANALYSIS_VERSION
    version = _ANALYSIS_SOURCE_VERSION + ":" + backend_key
    if version == ANALYSIS_VERSION:
        return
    ANALYSIS_VERSION = version
    _ANALYSIS_PAYLOADS.clear()
    _PREPARED_EXCHANGES.clear()
    _LOADED_ANALYSIS_VERSION = None
    for function in (exchange_analysis, reactive_act, turn_features, reaction_register,
                     cached_response_forms, _question_units):
        function.cache_clear()


def _remember_analysis(kind, text, payload):
    key = (kind, text)
    _ANALYSIS_PAYLOADS[key] = payload
    _ANALYSIS_PAYLOADS.move_to_end(key)
    while len(_ANALYSIS_PAYLOADS) > 65536:
        _ANALYSIS_PAYLOADS.popitem(last=False)


def _analysis_data(value):
    if is_dataclass(value):
        return {"class": type(value).__name__,
                "fields": {f.name: _analysis_data(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, (tuple, frozenset)):
        return {"tuple" if isinstance(value, tuple) else "frozenset":
                [_analysis_data(v) for v in value]}
    if isinstance(value, list):
        return [_analysis_data(v) for v in value]
    if isinstance(value, dict):
        return {k: _analysis_data(v) for k, v in value.items()}
    return value


@lru_cache(maxsize=32)
def _analysis_class(name):
    from debilgi.language import structured_utterance as ir
    cls = getattr(ir, name, None)
    if cls is None or not is_dataclass(cls):
        raise ValueError("unknown parser cache type")
    return cls


def _analysis_object(value):
    if "class" in value and "fields" in value:
        return _analysis_class(value["class"])(**value["fields"])
    if len(value) == 1:
        if "tuple" in value:
            return tuple(value["tuple"])
        if "frozenset" in value:
            return frozenset(value["frozenset"])
    return value


def _decode_analysis(payload):
    return json.loads(payload, object_hook=_analysis_object)


def _same_verb(left: str, right: str) -> bool:
    from debilgi.language.lexical_matching import verb_family

    return verb_family(left, right)


def ellipsis_variable(text: str) -> dict | None:
    """Closed interrogative grammar supplies a variable, never its value."""
    structure = analyze_structured_utterance(text)
    words = [t for t in structure.tokens if t.surface[:1].isalnum()]
    if len(words) > 5 or any(t.part_of_speech == "verb" for t in words):
        return None
    lemmas = {t.lemma for t in words}
    if {"потому", "что"} <= lemmas:
        return {"role": "cause_predicate", "relation": "cause",
                "subject": "ты" if "ты" in lemmas else None}
    if "почему" in lemmas or "что" in lemmas and "так" in lemmas:
        return {"role": "cause", "relation": "cause"}
    wh = next((t for t in words if t.lemma == "какой"), None)
    if wh:
        return {"role": "attribute", "features": {
            k: v for k, v in wh.grammatical_features if k in {"gender", "number", "case"}}}
    return None


def learned_evaluation(prompt, rows):
    """Independently witnessed human definitions can label an unfamiliar quality.

    Reply sentiment is not a definition. A suffix, morphological guess or a
    hundred repeats by one author cannot establish the sign of a new word.
    """
    source = exchange_analysis(prompt)[0]
    terms = {t.surface.casefold() for t in source.tokens
             if t.part_of_speech in {"noun", "adjective"}}
    observations = {}
    for row in (rows.evaluation_rows(terms) if isinstance(rows, _ExchangeRows) else rows):
        if not any(term in row["prompt"].casefold() for term in terms):
            continue
        structure = exchange_analysis(row["prompt"])[0]
        for clause in structure.clauses:
            if clause.negated:
                continue
            subjects = {structure.tokens[i].surface.casefold() for i in clause.subject_indices}
            demonstrative = next((structure.tokens[i] for i in clause.subject_indices
                                  if structure.tokens[i].lemma in {"это", "этот"}), None)
            if demonstrative:
                subjects.update(t.surface.casefold() for t in structure.tokens
                                if t.index < demonstrative.index
                                and t.semantic_role == "unassigned_content")
            if not terms & subjects:
                continue
            qualities = [structure.tokens[i] for i in (
                *clause.predicate_complement_indices,
                *((clause.predicate_index,) if clause.predicate_index is not None else ()))
                if structure.tokens[i].part_of_speech in {"adjective", "predicate", "adverb"}]
            values = [lexical_valence(t.surface) for t in qualities]
            values = [v for v in values if v is not None]
            if values:
                observations[row["prompt_event_id"]] = (
                    -1 if min(values) < 0 else 1, row["prompt_actor_id"])
    if len(observations) < 3 or len({p for _, p in observations.values()}) < 2:
        return None, ()
    signs = {s for s, _ in observations.values()}
    if len(signs) != 1:
        return None, tuple(observations)
    return float(next(iter(signs))), tuple(observations)


def evaluation_target(text):
    """A grammatical second person is a target; mere message addressing isn't."""
    structure = exchange_analysis(text)[0]
    subjects = [structure.tokens[i].lemma for c in structure.clauses
                for i in (*c.subject_indices, *c.experiencer_indices)]
    if subjects and all(s in {"ты", "вы"} for s in subjects) and not any(
        c.predicate_index is not None for c in structure.clauses
    ):
        return "addressee"
    return None


def learned_evaluation_exchange(prompt, rows):
    """Learn a quality's sign/direction from independent reciprocal appraisals.

    A laugh, a smile or a negative reply alone says nothing about the quality.
    A returned appraisal must have reciprocal grammar and a known sign. Surface
    families only retrieve witnesses; they never supply that sign themselves.
    """
    from debilgi.language.lexical_matching import expressive_family

    source = exchange_analysis(prompt)[0]
    terms = [t.surface for t in source.tokens if t.part_of_speech in {"noun", "adjective"}
             and t.semantic_role != "pragmatic_marker"]
    if any(c.subject_indices or c.experiencer_indices for c in source.clauses) \
            and evaluation_target(prompt) is None and sum(
                t.part_of_speech in {"noun", "adjective", "pronoun"} for t in source.tokens) > 1:
        return None, None, ()
    observations, directions = {}, {}
    for row in (rows.reciprocal_rows(terms) if isinstance(rows, _ExchangeRows) else rows):
        structure = exchange_analysis(row["prompt"])[0]
        if any(c.negated or c.predicate_index is not None for c in structure.clauses):
            continue
        subjects = [structure.tokens[i].lemma for c in structure.clauses
                    for i in (*c.subject_indices, *c.experiencer_indices)]
        if any(s not in {"ты", "вы"} for s in subjects) and sum(
            t.part_of_speech in {"noun", "adjective", "pronoun"} for t in structure.tokens
        ) > 1:
            continue  # third-person reports and self-descriptions don't target the listener
        if not any(expressive_family(term, t.surface) for term in terms for t in structure.tokens
                   if t.part_of_speech in {"noun", "adjective"}
                   and t.semantic_role != "pragmatic_marker"):
            continue
        answer = exchange_analysis(row["answer"])[0]
        if len(answer.clauses) != 1 or any(
            c.negated or c.predicate_index is not None for c in answer.clauses
        ) or not any(t.lemma == "сам" for t in answer.tokens):
            continue
        if any(
            answer.tokens[i].lemma not in {"ты", "вы"}
            for c in answer.clauses for i in (*c.subject_indices, *c.experiencer_indices)
        ):
            continue
        directions[row["prompt_event_id"]] = row["prompt_actor_id"], row["answer_event_id"]
        value = lexical_valence(row["answer"])
        if value is not None:
            observations[row["prompt_event_id"]] = (
                -1 if value < 0 else 1, row["prompt_actor_id"], row["answer_event_id"])
    sources = tuple(s for p, (_, a) in directions.items() for s in (p, a))
    target = ("addressee" if len(directions) >= 3
              and len({p for p, _ in directions.values()}) >= 2 else None)
    if len(observations) < 3 or len({p for _, p, _ in observations.values()}) < 2:
        return None, target, sources
    signs = {s for s, _, _ in observations.values()}
    if len(signs) != 1:
        return None, target, sources
    return float(next(iter(signs))), target, sources


def evaluation_tone_forms(prompt, rows, *, reaction="evaluation", recovery=False):
    """Transfer only a witnessed, nonassertive hearing reaction to an unknown evaluation.

    Independent uses across different turn types license the speech act, not
    an insult interpretation. No sentiment, agreement, laughter or content
    payload is transferred. The ordinary rule/feedback and reader checks apply.
    """
    groups = rows.tones() if isinstance(rows, _ExchangeRows) else {}
    for row in (() if isinstance(rows, _ExchangeRows) else rows):
        for form in cached_response_forms(row["prompt"], row["answer"])[0]:
            if (form.get("component") != "core" or form.get("meaning") != "acknowledge"
                    or form.get("response_turn") != "fragment"
                    or not all(s["part"] == "interjection" and s["binding"] == "speech_act"
                               for s in form["slots"])
                    or lexical_valence(" ".join(s["surface"] for s in form["slots"])) is not None):
                continue
            key = json.dumps(form["slots"], sort_keys=True)
            groups.setdefault(key, []).append((row, form))
    result = []
    for key, witnesses in sorted(groups.items()):
        people = {r["prompt_actor_id"] for r, _ in witnesses}
        turns = {reactive_act(r["prompt"]) or analyze_turn_form(r["prompt"]).kind
                 for r, _ in witnesses}
        if len({r["answer_event_id"] for r, _ in witnesses}) < 3 or len(people) < 2 \
                or len(turns) < 2:
            continue
        form = witnesses[0][1]
        sources = tuple(dict.fromkeys(s for r, _ in witnesses
                                     for s in (r["prompt_event_id"], r["answer_event_id"])))
        identity = ("evaluation_tone:" + key if reaction == "evaluation" and not recovery
                    else f"{reaction}_tone:recovery:" + key)
        identifier = "exchange-rule:" + hashlib.sha256(identity.encode()).hexdigest()[:24]
        result.append(ObservedDialogueForm(
            tuple(form["slots"]), "fragment", "evaluation", "fragment", sources,
            len(witnesses), 0.3, reactive_act=reaction, meaning="acknowledge",
            pattern_id=identifier.replace("exchange-rule:", "exchange-part:"),
            construction_id=form["construction_id"],
            construction_roles=tuple(f"act:{i}" for i in range(len(form["slots"]))),
            rule_id=identifier, rule_weight=0.7, activation=0.41 if recovery else 0.3,
            input_features=tuple(sorted(turn_features(prompt))),
            target_features=("meaning:acknowledge", "clause:fragment"),
            activation_source_ids=sources, mixture=(("reaction", 1.0),),
            recovery_only=recovery,
        ))
    return tuple(result)


def heard_question_forms(event, prompt, rows):
    """A witnessed noun echo can clarify a heard quality without asserting its sign."""
    from dataclasses import replace

    structure = exchange_analysis(prompt)[0]
    words = [t for t in structure.tokens if t.part_of_speech in {"noun", "adjective"}
             and t.semantic_role not in {"pragmatic_marker", "question_target", "function_word"}]
    if len(words) != 1:
        return ()
    for row in rows:
        raw = cached_response_forms(row["prompt"], row["answer"])[0]
        form = next((f for f in raw if f.get("component") == "core"
                     and f["clause_kind"] == "question" and len(f["slots"]) == 1
                     and f["slots"][0]["binding"] == "prompt"
                     and f["slots"][0]["part"] == "noun"), None)
        mark = next((f for f in raw if f.get("component") == "suffix"
                     and f["clause_kind"] == "question"
                     and any(s["surface"] == "?" for s in f["slots"])), None)
        if form is None or mark is None:
            continue
        slot = dict(form["slots"][0], source_role="heard_surface", features={})
        identifier = "exchange-rule:" + hashlib.sha256(
            ("heard_echo:" + form["construction_id"]).encode()).hexdigest()[:24]
        core = ObservedDialogueForm(
            (slot,), "question", "evaluation", "question",
            (row["prompt_event_id"], row["answer_event_id"], event.event_id), 1, .3,
            reactive_act="evaluation", pattern_id=identifier.replace("rule", "part"),
            meaning="slot_question", construction_id=form["construction_id"],
            construction_roles=("act:0",), rule_id=identifier, rule_weight=.7, activation=.41,
            input_features=tuple(sorted(turn_features(prompt))),
            target_features=("meaning:slot_question", "clause:question"),
            known_slots=(("heard_surface", words[0].surface, event.event_id),),
            mixture=(("reaction", 1.0),), recovery_only=True)
        suffix = replace(core, slots=tuple(mark["slots"]), component="suffix", meaning="tone",
                         pattern_id=core.pattern_id + ":mark", recovery_only=False,
                         construction_id=mark["construction_id"],
                         construction_roles=tuple(f"act:{i}" for i in range(len(mark["slots"]))))
        return core, suffix
    return ()


@dataclass(frozen=True, slots=True)
class ObservedDialogueForm:
    slots: tuple[dict, ...]
    clause_kind: str
    prompt_act: str
    response_act: str
    evidence_ids: tuple[str, ...]
    support: int
    relevance: float
    reactive_act: str | None = None
    pattern_id: str | None = None
    register: tuple[str, ...] = ()
    meaning: str | None = None
    component: str = "core"
    construction_id: str | None = None
    construction_roles: tuple[str, ...] = ()
    event_valence: float | None = None
    source_valence: float | None = None
    condition_report: bool = False
    rule_id: str | None = None
    rule_weight: float = 0.0
    activation: float = 0.0
    input_features: tuple[str, ...] = ()
    target_features: tuple[str, ...] = ()
    exceptions: tuple[str, ...] = ()
    activation_source_ids: tuple[str, ...] = ()
    hypothesis: tuple[tuple[str, str], ...] = ()
    known_slots: tuple[tuple[str, str, str], ...] = ()
    mixture: tuple[tuple[str, float], ...] = ()
    recovery_only: bool = False

    @property
    def confidence_level(self) -> str:
        # Acknowledging explicit uptake asserts reception, not a world hypothesis.
        received = self.meaning == "acknowledge" and self.reactive_act == "acknowledgment"
        return ("answer" if not self.hypothesis and (received or self.activation >= .40)
                else "hypothesis")


@lru_cache(maxsize=16384)
def turn_features(text: str) -> frozenset[str]:
    """Describe grammar and event tone independently of a candidate answer."""
    payload = _ANALYSIS_PAYLOADS.get(("features", text))
    if payload is not None:
        return _decode_analysis(payload)
    structure, _, _ = exchange_analysis(text)
    form = analyze_turn_form(text)
    from debilgi.language.canonical_claims import RELATIVE_DAYS

    result = {"turn:" + (reactive_act(text) or form.kind),
              *("tone:" + t for t in form.tone)}
    for token in structure.tokens:
        if token.lemma in RELATIVE_DAYS:
            result.add("time:" + str(RELATIVE_DAYS[token.lemma]))
        if token.semantic_role in {"subject", "experiencer"}:
            result.add("person:" + ({"я": "first", "мы": "first", "ты": "second",
                                    "вы": "second"}.get(token.lemma, "third")))
        if token.part_of_speech == "verb":
            result.update(k + ":" + v for k, v in token.grammatical_features
                          if k in {"tense", "transitivity", "form"})
            result.add("predicate:" + token.lemma)
        if token.semantic_role not in {"punctuation", "function_word", "pragmatic_marker"}:
            result.add("slot:" + token.semantic_role)
        if token.part_of_speech == "noun":
            result.add("topic:" + token.lemma)
            result.update("slot:" + token.semantic_role + ":" + k + ":" + v
                          for k, v in token.grammatical_features
                          if k in {"gender", "number", "case", "animacy"})
    for clause in structure.clauses:
        if clause.implicit_subject == "speaker":
            result.add("person:first")
        result.add("negated:" + str(clause.negated))
    value = lexical_valence(text)
    if value is not None:
        result.add("valence:" + ("negative" if value < 0 else "positive"))
    return frozenset(result)


def transformation_rule(prompt: str, form: dict) -> dict:
    """Store a feature difference on the existing human exchange edge.

    No topic lemma or answer string enters rule identity. Morphological content
    variables are bound later to the current input, including unfamiliar words.
    """
    features = turn_features(prompt)
    bound = any(s["binding"] == "prompt" or s.get("agreement_role") for s in form["slots"])
    bindings = {s.get("source_role") for s in form["slots"] if s["binding"] == "prompt"}
    bindings.update(s["agreement_role"] for s in form["slots"] if s.get("agreement_role"))
    if "topic" in bindings:
        bindings.update(t.semantic_role for t in exchange_analysis(prompt)[0].tokens
                        if t.part_of_speech == "noun")
    required = sorted(f for f in features if not f.startswith(("predicate:", "topic:"))
                      and not any(marker in f for marker in
                                  (":gender:", ":number:", ":case:", ":animacy:"))
                      and (not f.startswith("slot:") or f[5:] in bindings)
                      and (bound or f.startswith(("turn:", "tone:", "valence:"))
                           or "turn:self_report" in features
                           and f.startswith(("person:", "tense:"))))
    target = sorted({"meaning:" + form["meaning"], "clause:" + form["clause_kind"],
                     *("person:" + s["features"]["person"] for s in form["slots"]
                       if s["features"].get("person")),
                     *("bind:" + s["source_role"] for s in form["slots"]
                       if s.get("source_role"))})
    if form["meaning"] == "mirror_amusement":
        required = ["turn:amusement"]
    identity = json.dumps((required, target), sort_keys=True)
    return {"kind": "feature_transformation", "version": 1,
            "rule_id": "exchange-rule:" + hashlib.sha256(identity.encode()).hexdigest()[:24],
            "input": required, "target": target,
            "removed": sorted(set(required) - set(target)),
            "added": sorted(set(target) - set(required)),
            "response_turn": form.get("response_turn"),
            "anchors": sorted(f for f in features if f.startswith("predicate:")),
            "weight": 0.8, "support": 1, "exceptions": [], "feedback_ids": [],
            "feedback_targets": []}


def activate_transformation(rule, current, *, bridge=False, memory=None):
    from debilgi.language.semantic_graph import spread_activation

    required = set(rule["input"]) - {
        "tone:emphatic", "tone:smile", "tone:sad", "tone:caps",
    }
    signature = "|".join(sorted(current))
    if signature in rule.get("exceptions", ()):
        return 0.0
    turn = next((f for f in required if f.startswith("turn:")), "")
    same_turn = turn in current
    if not same_turn and not bridge:
        return 0.0
    bridge_strength = 1.0
    if not same_turn and bridge:
        current = set(current) | {turn}
        bridge_strength = 0.45
    # Person, negation and event valence are hard constraints, not votes.
    for prefix in ("person:", "negated:", "valence:"):
        expected = {f for f in required if f.startswith(prefix)}
        actual = {f for f in current if f.startswith(prefix)}
        if expected and (not actual or not expected & actual):
            return 0.0
    if not {f for f in required if f.startswith("tone:")} <= set(current):
        return 0.0
    arguments = {f for f in required if f in {
        "slot:direct_object", "slot:indirect_object", "slot:experiencer",
        "slot:action_complement", "slot:predicate_complement",
    } or f.startswith("slot:complement:")}
    if not arguments <= set(current):
        return 0.0
    match = len(required & current) / max(1, len(required))
    match *= bridge_strength
    if memory:
        match = min(1.0, match + 0.15 * max(
            (memory.get(a, 0.0) for a in rule.get("anchors", ())), default=0.0))
    # Conjunction gate prevents one shared particle from firing an entire rule.
    if match < 0.35:
        return 0.0
    identifier = rule["rule_id"]
    edges = [(f, identifier, match / max(1, len(required & current)))
             for f in required & current]
    edges.append((identifier, "target", float(rule["weight"])))
    return spread_activation({f: 1.0 for f in required & current}, edges).get("target", 0.0)


@lru_cache(maxsize=32768)
def exchange_analysis(text: str):
    payload = _ANALYSIS_PAYLOADS.get(("exchange", text))
    if payload is not None:
        return _decode_analysis(payload)
    from debilgi.language.clarification_policy import is_contact_request

    form = analyze_turn_form(text)
    structure = analyze_structured_utterance(text)
    pragmatic = analyze_pragmatics(text)
    act = analyze_dialogue_act(structure)
    # Social acts are grammatical/pragmatic categories, not response wording.
    kind = (
        "greeting"
        if ("greeting" in pragmatic.functions or is_contact_request(text))
        and not any(
            c.predicate_index is not None or c.clause_kind in {"question", "command"}
            for c in structure.clauses
        )
        else act.kind
    )
    features = frozenset(
        {
            f"lemma:{token.lemma}"
            for token in structure.tokens
            if token.surface[:1].isalnum() and token.part_of_speech not in {"particle"}
        }
        | {
            f"role:{t.semantic_role}:{t.part_of_speech}"
            for t in structure.tokens
            if t.part_of_speech is not None
        }
        | {f"clause:{c.clause_kind}" for c in structure.clauses}
        | {f"turn:{form.kind}", *(f"tone:{t}" for t in form.tone)}
    )
    return structure, kind, features


@lru_cache(maxsize=16384)
def reactive_act(text: str) -> str | None:
    """Describe a turn's social function, without choosing response words."""
    payload = _ANALYSIS_PAYLOADS.get(("reaction", text))
    if payload is not None:
        return json.loads(payload)
    form = analyze_turn_form(text)
    structure, kind, _ = exchange_analysis(text)
    if kind == "greeting" or form.kind in {"call", "greeting"}:
        return "greeting"
    if (len(structure.clauses) == 1 and structure.clauses[0].construction == "state_inquiry"
            and not any(t.lemma not in {"ты", "вы"} and t.part_of_speech == "pronoun"
                        for t in structure.tokens)):
        return "smalltalk"
    if any(c.clause_kind in {"question", "command"} for c in structure.clauses):
        return None
    from debilgi.social.signals import AFFIRMATION

    lexical = [t for t in structure.tokens if t.semantic_role != "punctuation"]
    if any(t.lemma in AFFIRMATION | {"окей"} for t in lexical) and not any(
        t.part_of_speech in {"noun", "verb", "adjective"} for t in lexical):
        return "acknowledgment"

    if (any(t.lemma == "я" and t.semantic_role in {"subject", "experiencer"}
            for t in structure.tokens) or any(c.implicit_subject == "speaker"
                                            for c in structure.clauses)):
        return "self_report"
    if kind in {"positive_evaluation", "negative_evaluation"}:
        return kind
    if form.kind == "fragment":
        from debilgi.language.dialogue_acts import _EVALUATIVE_QUALITY_LEMMAS
        from debilgi.language.lexical_matching import expressive_family

        if any(expressive_family(t.surface, quality) for t in structure.tokens
               if t.part_of_speech in {"noun", "adjective"}
               for quality in _EVALUATIVE_QUALITY_LEMMAS):
            return "evaluation"
    if any(c.implicit_predicate == "быть" and any(
        structure.tokens[i].lemma in {"ты", "вы"} for i in c.subject_indices
    ) for c in structure.clauses):
        for t in structure.tokens:
            if t.semantic_role != "predicate_complement":
                continue
            # Transfer existing social evidence through adjective morphology;
            # no new sentiment words or response phrases are introduced.
            quality = DEFAULT_MORPHOLOGY.inflect(
                t.lemma, form="short_adjective", gender="neuter", number="singular",
            )
            if quality:
                quality_kind = exchange_analysis(quality)[1]
                if quality_kind in {"positive_evaluation", "negative_evaluation"}:
                    return quality_kind
        return "evaluation"
    if "amusement" in analyze_pragmatics(text).functions:
        return "amusement"
    lexical = [t for t in structure.tokens if t.semantic_role != "punctuation"]
    if len(lexical) == 1 and any(
        "Prdx" in r.grammemes or r.part_of_speech == "predicate"
        for r in DEFAULT_MORPHOLOGY.parse_word(lexical[0].surface)
    ):
        return "emotion"
    return None


@lru_cache(maxsize=16384)
def reaction_register(text: str) -> tuple[str, ...]:
    pragmatic = analyze_pragmatics(text)
    return tuple(sorted(
        ({"coarse"} if "coarse_emphasis" in pragmatic.functions else set())
        | set(analyze_turn_form(text).tone)
        | ({"emphatic"} if "!" in text else set())
        | ({"formal"} if any(t.lemma == "вы" for t in exchange_analysis(text)[0].tokens)
           or "greeting" in pragmatic.functions and any(
               {"plur", "impr"} <= r.grammemes
               for t in exchange_analysis(text)[0].tokens
               for r in DEFAULT_MORPHOLOGY.parse_word(t.surface)) else set())
    ))


def _reaction_form(prompt: str, answer: str) -> dict | None:
    """Extract a non-referential act or a question bound to this prompt.

    Finite autobiographical clauses, locations, possessions and free content
    questions remain open. An evaluative fragment is a move, not a biography.
    Punctuation and slang come from the demonstration, including repeated ')'.
    """
    source, kind, _ = exchange_analysis(prompt)
    target, _, _ = exchange_analysis(answer)
    act = reactive_act(prompt) or analyze_turn_form(prompt).kind
    words = [t for t in target.tokens if t.surface[:1].isalnum()]
    if not words or len(words) > 7:
        return None
    question = any(c.clause_kind == "question" for c in target.clauses)
    agreement_words = [t for t in words if t.part_of_speech != "interjection"]
    agreement_fragment = (act == "positive_evaluation" and len(agreement_words) >= 2
                          and all(t.part_of_speech in {"particle", "conjunction"}
                                  for t in agreement_words))
    if (not question and words[-1].part_of_speech in {"conjunction", "preposition"}
            and not agreement_fragment):
        return None
    if not question and words[0].part_of_speech == "conjunction" and _COORDINATION_RELATIONS.get(
        words[0].lemma,
    ) not in {"additive", "contrast"}:
        return None
    if not question and all(t.part_of_speech in _CLOSED for t in words) and not (
        analyze_pragmatics(answer).functions & {
            "turn_management", "uncertainty", "hesitation", "emphatic_affirmation",
            "emphatic_rejection",
        } or any(evidence_counts(tuple(t.surface for t in words))[k]
                 for k in ("affirmation", "refusal"))
        or len(words) >= 2 and act in {"evaluation", "negative_evaluation", "fragment"}
        and any("Apro" in t.grammemes for t in words)
        or len(words) >= 2 and act == "positive_evaluation"
        and all(t.part_of_speech in {"particle", "conjunction"} for t in words)
    ):
        return None
    greetings = {cue.surface.casefold() for cue in analyze_pragmatics(answer).cues
                 if cue.function == "greeting"}
    greeting_end = max((t.index for t in words if t.surface.casefold() in greetings), default=-1)
    if act in {"greeting", "smalltalk"} and greeting_end >= 0 and any(
        t.index > greeting_end for t in words
    ):
        return None  # a greeting cannot license an attached state assertion
    if (act not in {"greeting", "smalltalk"}
            and any(cue.function == "greeting" and not any(
                t.surface.casefold() == cue.surface.casefold() and any(
                    "ADJS" in r.grammemes for r in DEFAULT_MORPHOLOGY.parse_word(t.surface)
                )
                for t in words
            ) for cue in analyze_pragmatics(answer).cues)):
        return None
    if act in {"positive_evaluation", "negative_evaluation", "evaluation"} and any(
        c.predicate_index is not None for c in source.clauses
    ):
        return None  # A report mixed with praise is not a pure evaluative demonstration.
    if len(target.clauses) != 1:
        return None
    source_lemmas = {t.lemma for t in source.tokens}
    if not question and any(c.modal_lemmas for c in target.clauses):
        return None
    if not question and any(
        t.lemma in {"ты", "вы", "он", "она", "они"}
        and t.semantic_role in {"subject", "experiencer"} for t in words
    ):
        return None
    if not question and any(t.part_of_speech == "predicate" or any(
        "Prdx" in r.grammemes for r in DEFAULT_MORPHOLOGY.parse_word(t.surface)
    ) for t in words) and any(
        t.part_of_speech == "pronoun" and t.lemma != "это" for t in words
    ):
        return None
    if question and not any(c.predicate_index is not None for c in target.clauses) and (
        len(words) > 4 or any(t.part_of_speech == "adverb"
                             and t.semantic_role not in {"question_target", "pragmatic_marker"}
                             and t.lemma not in source_lemmas
                             for t in words)
    ):
        return None
    epistemic = (not question and any(c.negated for c in target.clauses)
                 and any(t.lemma in {"знать", "помнить"} and t.part_of_speech == "verb"
                         for t in words)
                 and all(t.lemma in {"знать", "помнить", "не", "я"}
                         or t.part_of_speech in {"particle", "conjunction"}
                         or t.semantic_role == "pragmatic_marker" for t in words))
    if not epistemic and not question and (
        any(c.negated or c.clause_kind == "command" for c in target.clauses)
        or any(t.lemma in {"не", "нет"} for t in words)
    ):
        # A free discourse particle cannot invert the committed reaction act.

        return None
    if not epistemic and any(t.lemma in {"я", "мы", "мой", "наш"} for t in words):
        return None
    if not epistemic and any(t.part_of_speech == "verb" and t.lemma not in source_lemmas
                             for t in words):
        return None
    if (not question and not epistemic
            and any(c.predicate_index is not None
                    and target.tokens[c.predicate_index].part_of_speech != "predicate"
                    for c in target.clauses)):
        return None
    if not question and any(t.part_of_speech == "preposition" for t in words):
        return None
    if unbound_anaphors(target, source):
        # A pronoun that refers to nothing in the turn it answered is a fragment of
        # someone else's thought; it cannot be demonstrated as a reusable reaction.
        return None
    slots = []
    for token in target.tokens:
        slot = {"surface": token.surface, "lemma": token.lemma,
                "part": token.part_of_speech, "features": dict(token.grammatical_features)}
        if not token.surface[:1].isalnum():
            slot["binding"] = "punctuation"
        elif (token.lemma in source_lemmas and token.part_of_speech not in _CLOSED
              and token.semantic_role != "pragmatic_marker" and not any(
                  r.part_of_speech == "interjection"
                  for r in DEFAULT_MORPHOLOGY.parse_word(token.surface))):
            slot.update(binding="prompt", source_role=next(
                t.semantic_role for t in source.tokens if t.lemma == token.lemma),
            )
            slot["source_experiencer"] = any(c.experiencer_indices for c in source.clauses)
            if question and act == "evaluation" and len(words) == 1:
                slot["source_role"] = "heard_surface"
            if question and not any(r.is_known for r in DEFAULT_MORPHOLOGY.parse_word(
                token.surface,
            )):
                slot["source_role"] = "unfamiliar_word"
        elif epistemic and token.lemma in {"знать", "помнить"}:
            slot["binding"] = "speech_act"
        elif (token.part_of_speech in _CLOSED or token.semantic_role == "question_target"
              or "Apro" in token.grammemes):
            slot["binding"] = "grammar"
            if token.semantic_role == "question_target" and token.part_of_speech == "adjective":
                nouns = [t for t in source.tokens if t.part_of_speech == "noun"]
                if len(nouns) == 1:
                    slot["agreement_role"] = nouns[0].semantic_role
        elif any(c.clause_kind == "question" and c.token_start <= token.index < c.token_end
                 for c in target.clauses):
            if token.part_of_speech == "adverb" and any(
                t.part_of_speech == "verb" and t.lemma in source_lemmas for t in words
            ):
                slot["binding"] = "grammar"  # observed degree/manner, no new entity
            else:
                return None
        elif (token.part_of_speech == "interjection" or kind == "fragment"
              and token.lemma in source_lemmas and any(r.part_of_speech == "interjection"
                  for r in DEFAULT_MORPHOLOGY.parse_word(token.surface))) or (
            (token.part_of_speech == "adverb" or token.part_of_speech == "noun" and all(
                not r.is_known for r in DEFAULT_MORPHOLOGY.parse_word(token.surface)
            )) and any(
                "ADJS" in r.grammemes for r in DEFAULT_MORPHOLOGY.parse_word(token.surface)
            )
            or
            token.semantic_role == "pragmatic_marker"
            or exchange_analysis(token.surface)[1] in {"positive_evaluation", "negative_evaluation"}
            or any("Prdx" in r.grammemes for r in DEFAULT_MORPHOLOGY.parse_word(token.surface))
            or token.part_of_speech == "predicate"
            or token.part_of_speech == "adverb" and lexical_valence(answer) is not None
            or token.part_of_speech == "adverb" and any(
                "Prdx" in r.grammemes or r.part_of_speech == "predicate"
                for t in words if t.index != token.index
                for r in DEFAULT_MORPHOLOGY.parse_word(t.surface))
            or token.part_of_speech == "noun" and len(words) <= 2
            and lexical_valence(prompt) is not None and lexical_valence(prompt) < 0
            or (act in {"evaluation", "positive_evaluation", "negative_evaluation"}
                or kind == "fragment" and sum(t.surface[:1].isalnum() for t in source.tokens) == 1)
            and any("Apro" in t.grammemes and t.semantic_role == "unassigned_content"
                    for t in words)
        ):
            slot["binding"] = "speech_act"
        else:
            return None
        slots.append(slot)
    if (question and not any(s["binding"] == "prompt" or s.get("agreement_role") for s in slots)
            and (not any(t.lemma in {"ты", "вы"} for t in words) or any(
            t.part_of_speech == "particle" and t.semantic_role != "pragmatic_marker"
            or t.part_of_speech == "pronoun" and t.lemma not in {"ты", "вы"}
            and t.semantic_role != "question_target" for t in words
            ))):
        return None
    if (question and not any(s["binding"] == "prompt" or s.get("agreement_role") for s in slots)
            and not any(t.semantic_role == "question_target" or t.part_of_speech == "preposition"
                        for t in words)):
        return None
    if (act == "self_report" and not question and words[-1].part_of_speech == "conjunction"):
        return None
    if (act == "self_report" and not question and any(
        t.part_of_speech == "noun" and dict(t.grammatical_features).get("animacy") == "animate"
        and any(r.is_known for r in DEFAULT_MORPHOLOGY.parse_word(t.surface)) for t in words
    )):
        return None
    if not question and any(s["binding"] == "prompt" for s in slots):
        return None
    if question and act in {"evaluation", "positive_evaluation", "negative_evaluation"} and not any(
        s.get("source_role") in {"unfamiliar_word", "heard_surface"} for s in slots
    ):
        return None
    nouns = [t for t in source.tokens if t.part_of_speech == "noun"]
    bound_nouns = [s for s in slots if s["binding"] == "prompt" and s["part"] == "noun"]
    agreement_role = (bound_nouns[0]["source_role"] if len(bound_nouns) == 1
                      else nouns[0].semantic_role if len(nouns) == 1 else None)
    if question and agreement_role is not None:
        for slot in slots:
            if (slot["binding"] == "grammar" and slot["part"] == "adjective"
                    and any("Apro" in r.grammemes
                            for r in DEFAULT_MORPHOLOGY.parse_word(slot["surface"]))):
                slot["agreement_role"] = agreement_role
    return {"clause_kind": "question" if question else "fragment", "slots": slots,
            "response_turn": analyze_turn_form(answer).kind,
            "reactive_act": act or kind, "register": reaction_register(prompt),
            "prompt_negated": any(c.negated for c in source.clauses),
            "response_register": reaction_register(answer),
            "stimulus_valence": lexical_valence(prompt), "epistemic": epistemic,
            "question_slot": next((t.lemma for t in target.tokens
                                   if t.semantic_role == "question_target"), None)}


def response_forms(prompt: str, answer: str) -> tuple[list[dict], list[dict]]:
    source, _, _ = exchange_analysis(prompt)
    target, _, _ = exchange_analysis(answer)
    forms, links = [], []
    source_lemmas = {token.lemma for token in source.tokens}
    social = {
        cue.surface.casefold()
        for cue in analyze_pragmatics(answer).cues
        if cue.function == "greeting"
    }
    for clause in target.clauses:
        slots = []
        for token in target.tokens[clause.token_start : clause.token_end]:
            if not token.surface[:1].isalnum():
                continue
            features = dict(token.grammatical_features)
            slot = {"part": token.part_of_speech, "role": token.semantic_role, "features": features}
            if token.surface.casefold() in social:
                slot.update(binding="speech_act", lemma=token.lemma)
            elif token.part_of_speech in _CLOSED or token.semantic_role == "question_target":
                slot.update(binding="grammar", lemma=token.lemma, surface=token.surface.casefold())
            elif token.lemma in source_lemmas:
                source_token = next(t for t in source.tokens if t.lemma == token.lemma)
                slot.update(binding="prompt", source_role=source_token.semantic_role)
                links.append(
                    {
                        "prompt_role": source_token.semantic_role,
                        "response_role": token.semantic_role,
                        "part": token.part_of_speech,
                    }
                )
            else:
                slot.update(binding="open")
            slots.append(slot)
        if slots:
            forms.append({"clause_kind": clause.clause_kind, "slots": slots})
    forms.extend(_reaction_parts(prompt, answer))
    # Retain an explicit human appraisal with its original surface and
    # direction; a reply's popularity or sentiment alone cannot label a word.
    if reactive_act(prompt) in {"evaluation", "positive_evaluation", "negative_evaluation"}:
        for token in source.tokens:
            if token.part_of_speech in {"noun", "adjective"}:
                links.append({"kind": "evaluation_observation", "surface": token.surface,
                              "lemma": token.lemma, "target": evaluation_target(prompt),
                              "sign": lexical_valence(prompt), "epistemic_status": "working",
                              "derivation_depth": 0})
    # Raw human evidence about unfamiliar vocabulary lives on its existing
    # exchange edge. It is never recursively re-observed as a semantic fact.
    for token in source.tokens:
        if token.surface[:1].isalpha() and not any(
            r.is_known for r in DEFAULT_MORPHOLOGY.parse_word(token.surface)
        ):
            form = analyze_turn_form(answer)
            links.append({"kind": "word_reaction", "lemma": token.lemma,
                          "surface": token.surface, "move": form.kind,
                          "tone": list(form.tone), "derivation_depth": 0})
    for form in forms:
        form["projection_version"] = 10
        if form.get("component") == "core":
            rule = transformation_rule(prompt, form)
            form["rule_id"] = rule["rule_id"]
            if not any(link.get("rule_id") == rule["rule_id"] for link in links):
                links.append(rule)
    return forms, links


def normalized_response(text: str) -> str:
    """Ignore typography as well as punctuation: a new smile is still a copy."""
    return " ".join(re.findall(r"\w+", text.casefold().replace("ё", "е")))


def response_similarity(text: str, recent: tuple[str, ...]) -> float:
    """One normalization for delivered-response repetition, separate from word reuse."""
    from difflib import SequenceMatcher

    normalized = normalized_response(text)
    return max((SequenceMatcher(None, normalized, normalized_response(old)).ratio()
                for old in recent), default=0.0)


def response_repetition(text: str, recent: tuple[str, ...]) -> float:
    """Exact own-response reuse decays with distance from the latest receipt."""
    normalized = normalized_response(text)
    return max((1 / (index + 1) for index, old in enumerate(recent)
                if normalized_response(old) == normalized), default=0.0)




def reaction_slot_tokens(structure, slot):
    """One shared binding policy for retrieval and the morphological composer."""
    role = slot.get("agreement_role") or slot.get("source_role")
    if role == "heard_surface":
        return [t for t in structure.tokens if t.part_of_speech in {"noun", "adjective"}
                and t.semantic_role not in {"pragmatic_marker", "question_target", "function_word"}]
    if role == "deictic_time":
        from debilgi.language.canonical_claims import RELATIVE_DAYS

        return [t for t in structure.tokens if t.lemma in RELATIVE_DAYS][:1]
    if role == "condition_word":
        return [t for t in structure.tokens if t.lemma == slot["lemma"]][:1]
    if role == "requested_predicate":
        return [t for t in structure.tokens if t.part_of_speech == "verb"][-1:]
    if (role == "predicate" and slot["part"] == "verb"
            and bool(slot.get("source_experiencer")) != any(
                c.experiencer_indices for c in structure.clauses)):
        return []
    if role == "predicate" and slot["part"] == "verb":
        # An analytic future (будешь молчать) names its event by the infinitive;
        # the auxiliary «быть» is not the predicate being asked about.
        events = list(analytic_future_events(structure).items())
        if events:
            hidden = {index for index, _ in events}
            chosen = [t for t in structure.tokens if t.semantic_role == role
                      and t.part_of_speech == "verb" and t.index not in hidden]
            return chosen + [action for _, action in events]
    if role == "unfamiliar_word":
        return [t for t in structure.tokens if t.surface[:1].isalpha() and not any(
            r.is_known for r in DEFAULT_MORPHOLOGY.parse_word(t.surface)
        )][-1:]
    return [t for t in structure.tokens if (
        role == "topic" and t.part_of_speech == "noun" and t.semantic_role not in {
            "pragmatic_marker", "vocative", "duration", "time"}
        or t.semantic_role == role
    ) and (slot.get("agreement_role") or t.part_of_speech == slot["part"])
        and not (slot["part"] == "verb" and slot["features"].get("transitivity")
                 and dict(t.grammatical_features).get("transitivity")
                 and slot["features"]["transitivity"]
                 != dict(t.grammatical_features)["transitivity"])]


def _reaction_parts(prompt: str, answer: str) -> list[dict]:
    """Project raw human clauses once into cores, tone and grammatical slots.

    Each clause is checked separately. A later question cannot license a
    preceding state assertion. Only smalltalk's independent reciprocal question
    can be recovered after an unsafe autobiographical clause.
    """
    target, _, _ = exchange_analysis(answer)
    act = reactive_act(prompt) or analyze_turn_form(prompt).kind
    result, leading = [], True
    for index, clause in enumerate(target.clauses):
        end = (target.clauses[index + 1].token_start
               if index + 1 < len(target.clauses) else len(target.tokens))
        tokens = target.tokens[clause.token_start:end]
        while tokens and not tokens[0].surface[:1].isalnum():
            tokens = tokens[1:]
        # Commas needn't create separate parser clauses. Isolate a leading
        # greeting before checking the remainder, including copular states.
        if leading and act in {"greeting", "smalltalk"}:
            greetings = {cue.surface.casefold() for cue in analyze_pragmatics(answer).cues
                         if cue.function == "greeting"}
            greeting_end = max((i for i, t in enumerate(tokens)
                                if t.surface.casefold() in greetings), default=-1)
            if greeting_end >= 0 and any(t.surface[:1].isalnum()
                                        for t in tokens[greeting_end + 1:]):
                tokens = tokens[:greeting_end + 1]
        text = re.sub(r"\s+([,.!?;:)])", r"\1", " ".join(t.surface for t in tokens))
        # An independent epistemic head can be reused; its following guessed
        # answer and biography cannot. Preserve just the witnessed clause head.
        epistemic_end = next((i for i, t in enumerate(tokens)
                              if t.part_of_speech == "verb" and t.lemma in {"знать", "помнить"}
                              and any(p.lemma == "не" for p in tokens[:i])), None)
        if epistemic_end is not None and any(
            t.surface in {",", ".", "…"} for t in tokens[epistemic_end + 1:epistemic_end + 2]
        ):
            text = " ".join(t.surface for t in tokens[:epistemic_end + 1])
        form = _reaction_form(prompt, text)
        if form is None:
            leading = False
            continue
        if not leading and not (act == "smalltalk" and form["clause_kind"] == "question"):
            continue
        slots = list(form["slots"])
        marks = []
        while slots and slots[-1]["binding"] == "punctuation":
            marks.insert(0, slots.pop())
        if not slots:
            continue
        question = form["clause_kind"] == "question"
        if form.get("epistemic"):
            meaning = "express_limit"
        elif question:
            meaning = "slot_question" if any(
                s["binding"] == "prompt" or s.get("agreement_role") for s in slots
            ) else "reciprocal_question"
        elif any(c.function == "greeting" for c in analyze_pragmatics(text).cues) and act in {
            "greeting", "smalltalk",
        }:
            meaning = "return_greeting"
        elif act in {"negative_evaluation", "evaluation", "fragment"} and any(
            s["lemma"] in {"ты", "вы", "твой", "ваш", "сам", "сама"}
            for s in slots
        ):
            meaning = "retort"
        elif act == "positive_evaluation" and any(
            s["part"] in {"particle", "conjunction"} for s in slots
        ) and all(s["part"] in {"particle", "conjunction", "interjection"} for s in slots):
            meaning = "agree"
        elif (exchange_analysis(text)[1] in {"positive_evaluation", "negative_evaluation"}
              and any(s["part"] not in _CLOSED | {"interjection"} for s in slots)
              or any("ADJS" in r.grammemes for s in slots
                     for r in DEFAULT_MORPHOLOGY.parse_word(s["surface"]))):
            meaning = "appraise"
        elif "amusement" in analyze_pragmatics(text).functions:
            meaning = "mirror_amusement"
        elif all(s["part"] == "interjection" or s["binding"] == "grammar" for s in slots):
            meaning = "acknowledge"
        elif act == "self_report" and lexical_valence(prompt) is not None \
                and lexical_valence(prompt) < 0:
            meaning = "empathize"
        elif act == "amusement":
            meaning = "mirror_amusement"
        elif any(s["part"] == "adverb" for s in slots):
            # Qualitative adverbs and valenced words were read as appraisals
            # above. A remaining adverb is a circumstance of the donor's own
            # content (place, time, degree: «рядом», «потом»), an answer to the
            # donor's question rather than a transferable evaluation. It is
            # neither an appraisal nor a hearing receipt.
            leading = False
            continue
        else:
            meaning = "acknowledge"

        if meaning == "acknowledge" and any(
            s["part"] == "predicate" for s in slots
        ):
            continue  # an unbound existence/state assertion is not a hearing receipt

        if act == "self_report" and lexical_valence(text) is not None \
                and lexical_valence(text) < 0:
            meaning = "empathize"
        for slot in slots:
            if slot.get("agreement_role"):
                nouns = [t for t in exchange_analysis(prompt)[0].tokens
                         if t.part_of_speech == "noun" and t.semantic_role not in {
                             "pragmatic_marker", "vocative", "duration", "time"}]
                if len(nouns) == 1:
                    slot["agreement_role"] = "topic"

        def add(component, selected, semantic=meaning, base=form):
            if selected:
                shape = (component, base["clause_kind"], tuple(
                    (s["binding"], s["part"], s.get("source_role"), s.get("agreement_role"),
                     tuple(sorted(s["features"].items()))) for s in selected))
                construction = "exchange-syntax:" + hashlib.sha256(
                    json.dumps(shape, sort_keys=True).encode()).hexdigest()[:24]
                result.append({**base, "slots": selected, "meaning": semantic,
                               "component": component, "projection_version": 6,
                               "construction_id": construction})

        # A tone prefix is a demonstrated pragmatic token, never an authored
        # connective. It can combine with a different observed core.
        if len(slots) > 1 and slots[0]["lemma"] not in {"не", "ни", "нет"} and (
                              "coarse" in reaction_register(slots[0]["surface"])
                              or slots[0]["part"] == "conjunction" and meaning != "agree"
                              and _COORDINATION_RELATIONS.get(slots[0]["lemma"])
                              in {"additive", "contrast"}
                              or slots[0]["part"] == "particle" and exchange_analysis(
                                  slots[0]["surface"])[0].tokens[0].semantic_role
                                  == "pragmatic_marker"
                              or (
            slots[0]["part"] == "interjection" and (
                slots[1]["part"] == "adverb" or meaning == "agree")
        )) and (
            exchange_analysis(slots[0]["surface"])[0].tokens[0].semantic_role
            in {"pragmatic_marker", "function_word"}
        ):
            prefix = slots.pop(0)
            add("prefix", [prefix], "tone")
            if prefix["part"] == "interjection" and "coarse" not in reaction_register(
                prefix["surface"],
            ):
                add("core", [prefix], "acknowledge")
        if meaning == "retort":
            payload = [s for s in slots if s["binding"] == "speech_act"
                       and s["part"] in {"noun", "adjective"} and not any(
                           "Apro" in r.grammemes
                           for r in DEFAULT_MORPHOLOGY.parse_word(s["surface"]))]
            add("payload", payload)
            slots = [s for s in slots if s not in payload]
        # Repetition supplies emphasis; its word is learned once as the core.
        if slots and all(s["lemma"] == slots[0]["lemma"] for s in slots):
            slots = slots[:1]
        add("core", slots)
        if all(s["surface"] in {".", "!", "?", "…", ")", "("} for s in marks):
            add("suffix", marks, "tone")
    return result


async def _store_part_constructions(connection, guild, records, timestamp, *, increment=False):
    """Existing construction memory holds abstract role orders, never answers."""
    for identifier, (form, count) in records.items():
        roles = tuple(f"act:{i}" for i in range(len(form["slots"])))
        update = ("observation_count + excluded.observation_count" if increment
                  else "MAX(observation_count, excluded.observation_count)")
        await connection.execute(
            """INSERT INTO language_productive_constructions
               (guild_id, construction_id, clause_kind, predicate_class, roles_json,
                required_roles_json, observation_count, confidence, source, created_at, updated_at)
               VALUES (?, ?, ?, 'dialogue_act', ?, ?, ?, 0.75, 'observed_exchange_parts', ?, ?)
               ON CONFLICT(guild_id, construction_id) DO UPDATE SET
                 observation_count = """ + update + ", updated_at = excluded.updated_at",
            (guild, identifier, form["clause_kind"], json.dumps(roles), json.dumps(roles),
             count, timestamp, timestamp),
        )


class _ExchangeRows(list):
    """Ordered exchange projection with reversible per-exchange contributions.

    Inserts, window eviction and feedback update only changed contributors.
    Source order and distinct witness thresholds are preserved by consumers.
    """
    def __init__(self):
        super().__init__()
        self.records = {}
        self.tone_groups = {}
        self.definitions = {}
        self.reciprocals = {}
        self.texts = {}
        self.form_groups = {}
        self.rule_contributions = {}
        self.construction_contributions = {}
        self._totals = None
        self._questions = None
        self._time_edge = None
        self._time_ready = False

    def update(self, rows):
        keys = {r["answer_event_id"] for r in rows}
        changed = set(self.records) - keys
        for row in rows:
            old = self.records.get(row["answer_event_id"])
            if old is None or any(old[k] != row[k] for k in (
                "prompt", "answer", "response_forms_json", "topic_links_json")):
                changed.add(row["answer_event_id"])
        if self and not changed and len(rows) == len(self) and all(
            self.order[row["answer_event_id"]] == i for i, row in enumerate(rows)
        ):
            # A cache miss in the async reader need not change this projection.
            # Keep the existing question graph and aggregates for equal inputs.
            return
        for identifier in changed:
            old = self.records.pop(identifier, None)
            if old is None:
                continue
            for groups in (self.tone_groups, self.definitions, self.reciprocals,
                           self.form_groups, self.rule_contributions,
                           self.construction_contributions):
                for key in list(groups):
                    groups[key].pop(identifier, None)
                    if not groups[key]:
                        del groups[key]
        for raw in rows:
            identifier = raw["answer_event_id"]
            if identifier in self.records:
                continue
            row = self._prepared(raw)
            self.records[identifier] = row
            stored = row["_stored"]
            for rule_id, rule in row["_rules"].items():
                self.rule_contributions.setdefault(rule_id, {})[identifier] = rule
            for construction_id in {f["construction_id"] for f in stored
                                    if f.get("construction_id")}:
                form = next(f for f in stored if f.get("construction_id") == construction_id)
                self.construction_contributions.setdefault(construction_id, {})[identifier] = form
            if stored:
                pattern = (row.get("prompt_act"), row.get("response_act"))
                self.form_groups.setdefault(pattern, {})[identifier] = row
            source = row["_analysis"][0]
            for clause in source.clauses:
                subjects = {source.tokens[i].surface.casefold() for i in clause.subject_indices}
                demo = next((source.tokens[i] for i in clause.subject_indices
                             if source.tokens[i].lemma in {"это", "этот"}), None)
                if demo:
                    subjects.update(t.surface.casefold() for t in source.tokens
                                    if t.index < demo.index
                                    and t.semantic_role == "unassigned_content")
                if not clause.negated:
                    for term in subjects:
                        self.definitions.setdefault(term, {})[identifier] = row
            if not any(c.negated or c.predicate_index is not None for c in source.clauses):
                answer = exchange_analysis(row["answer"])[0]
                if len(answer.clauses) == 1 and not any(
                    c.negated or c.predicate_index is not None for c in answer.clauses
                ) and any(t.lemma == "сам" for t in answer.tokens):
                    for token in source.tokens:
                        if token.part_of_speech in {"noun", "adjective"}:
                            self.reciprocals.setdefault(token.surface, {})[identifier] = row
            for form in cached_response_forms(row["prompt"], row["answer"])[0]:
                if (form.get("component") == "core" and form.get("meaning") == "acknowledge"
                    and form.get("response_turn") == "fragment"
                    and all(t["part"] == "interjection" and t["binding"] == "speech_act"
                            for t in form["slots"])
                    and lexical_valence(" ".join(t["surface"] for t in form["slots"])) is None):
                    key = json.dumps(form["slots"], sort_keys=True)
                    entries = self.tone_groups.setdefault(key, {}).setdefault(identifier, [])
                    entries.append((row, form))
        self[:] = [self.records[r["answer_event_id"]] for r in rows]
        self.order = {r["answer_event_id"]: i for i, r in enumerate(self)}
        self._questions = None
        self._time_ready = False
        self._totals = None

    @staticmethod
    def _prepared(raw):
        key = tuple(sorted(dict(raw).items()))
        cached = _PREPARED_EXCHANGES.get(key)
        if cached is not None:
            _PREPARED_EXCHANGES.move_to_end(key)
            return cached
        row = dict(raw)
        row["_stored"] = json.loads(row["response_forms_json"])
        row["_links"] = json.loads(row["topic_links_json"])
        row["_upgrade"] = None
        stored = row["_stored"]
        # Upgrade the existing projection, without replaying evidence or
        # increasing support. Both fresh seed and chat use response_forms.
        links = row["_links"]
        if not stored or any(f.get("projection_version") != 10 or (
            f.get("meaning") and not f.get("construction_id")) for f in stored):
            old_rules = {link.get("rule_id"): link for link in links if link.get("rule_id")}
            stored, links = cached_response_forms(row["prompt"], row["answer"])
            links = [dict(link, **{k: old_rules[link["rule_id"]][k] for k in (
                "weight", "support", "exceptions", "feedback_ids", "feedback_targets",
            ) if k in old_rules[link["rule_id"]]}) if link.get("rule_id") in old_rules
                else link for link in links]
            links += [link for link in old_rules.values() if link.get("projection_operator")]
            links = [dict(link, source_ids=[row["prompt_event_id"], row["answer_event_id"]])
                     if link.get("kind") == "feature_transformation" else link
                     for link in links]
            form_json = json.dumps(stored, ensure_ascii=False, sort_keys=True)
            link_json = json.dumps(links, ensure_ascii=False, sort_keys=True)
            if (form_json, link_json) != (row["response_forms_json"], row["topic_links_json"]):
                row["_upgrade"] = (form_json, link_json, row["answer_event_id"])
        row["_stored"], row["_links"] = stored, links
        row["_rules"] = {link["rule_id"]: link for link in links
                         if link.get("kind") == "feature_transformation"}
        row["_analysis"] = exchange_analysis(row["prompt"])
        row["_reaction"] = reactive_act(row["prompt"]) or next(
            feature[5:] for feature in row["_analysis"][2] if feature.startswith("turn:"))
        row["_content"] = frozenset(t.lemma for t in row["_analysis"][0].tokens
            if t.part_of_speech in {"noun", "verb", "adjective"}
            and t.semantic_role not in {
                "pragmatic_marker", "question_target", "time", "duration"})
        _PREPARED_EXCHANGES[key] = row
        _PREPARED_EXCHANGES.move_to_end(key)
        while len(_PREPARED_EXCHANGES) > 16384:
            _PREPARED_EXCHANGES.popitem(last=False)
        return row

    def ordered(self, candidates):
        return sorted(candidates.values(), key=lambda r: self.order[r["answer_event_id"]])

    def evaluation_rows(self, terms):
        return self.ordered({i: r for term in terms
                             for i, r in self.definitions.get(term, {}).items()})

    def reciprocal_rows(self, terms):
        from debilgi.language.lexical_matching import expressive_family
        return self.ordered({i: r for word, entries in self.reciprocals.items()
                             if any(expressive_family(term, word) for term in terms)
                             for i, r in entries.items()})

    def tones(self):
        return {key: [item for identifier in sorted(entries, key=self.order.__getitem__)
                      for item in entries[identifier]] for key, entries in self.tone_groups.items()}

    def form_rows(self, patterns):
        return self.ordered({i: row for pattern in patterns
                             for i, row in self.form_groups.get(pattern, {}).items()})

    def projection_totals(self):
        if self._totals is None:
            rules = {}
            for identifier, entries in self.rule_contributions.items():
                ordered = [entries[i] for i in sorted(entries, key=self.order.__getitem__)]
                rules[identifier] = (sum(r.get("support", 1) for r in ordered),
                                     [r["weight"] for r in ordered])
            constructions = {identifier: (entries[max(entries, key=self.order.__getitem__)],
                                           len(entries))
                             for identifier, entries in self.construction_contributions.items()}
            upgrades = [row["_upgrade"] for row in self if row["_upgrade"] is not None]
            self._totals = upgrades, constructions, rules
        return self._totals

    def question_projection(self):
        if self._questions is not None:
            return self._questions
        from debilgi.language.canonical_claims import RELATIVE_DAYS
        texts = {}
        for row in self:
            texts.setdefault(row["prompt_event_id"], row["prompt"])
            texts.setdefault(row["answer_event_id"], row["answer"])
        units, temporal, neighbours = [], None, {}
        for source_id, text in texts.items():
            # Per-text contribution survives new exchanges and window eviction.
            if text not in self.texts:
                source = exchange_analysis(text)[0]
                edges, times = [], []
                for clause in source.clauses:
                    tokens = source.tokens[clause.token_start:clause.token_end]
                    verbs = tuple(t.lemma for t in tokens if t.part_of_speech == "verb"
                        and t.lemma not in {"быть", "мочь", "уметь", "хотеть", "любить", "делать"}
                        and dict(t.grammatical_features).get("transitivity") == "intransitive")
                    edges.append((tuple(t.lemma for t in tokens), verbs))
                    if clause.clause_kind == "question" and clause.predicate_index is not None:
                        words = [t for t in tokens if t.lemma in RELATIVE_DAYS]
                        if words:
                            times.append("after" if words[0].index > clause.predicate_index
                                         else "before")
                self.texts[text] = (edges, times, _question_units("", text))
            edges, times, raw_units = self.texts[text]
            for lemmas, verbs in edges:
                for lemma in lemmas:
                    for verb in verbs:
                        neighbours.setdefault(lemma, {}).setdefault(verb, set()).add(source_id)
            for position in times:
                if temporal is None or position == "after":
                    temporal = (source_id, position)
            units.extend(dict(unit, source_id=source_id) for unit in raw_units)
        self._questions = units, temporal, neighbours
        return self._questions


class DialogueStructureLearning:
    def __init__(self, database: Database, assimilation) -> None:
        from debilgi.language.interlocutor_intent import InterlocutorIntentRepository

        self._database = database
        self._assimilation = assimilation
        self.interlocutor = InterlocutorIntentRepository(database)
        self._read_cache = {}
        self._exchange_revision = 0
        self._exchange_rows = {}
        self._analysis_loaded = None
        self._warm_task = None
        self._analysis_load_lock = asyncio.Lock()
        self._projection_jobs = {}

    def start_analysis_warmup(self):
        if self._warm_task is None or self._warm_task.done():
            self._warm_task = asyncio.create_task(self.warm_analysis_cache())
            self._database._maintenance_tasks.add(self._warm_task)
            self._warm_task.add_done_callback(self._database._maintenance_tasks.discard)

    async def stop_analysis_warmup(self):
        if self._warm_task is not None:
            self._warm_task.cancel()
            await asyncio.gather(self._warm_task, return_exceptions=True)
            self._warm_task = None

    async def _load_analysis_cache(self):
        async with self._analysis_load_lock:
            await self._hydrate_analysis_cache()

    async def _hydrate_analysis_cache(self):
        global _LOADED_ANALYSIS_VERSION
        if self._analysis_loaded == ANALYSIS_VERSION:
            return
        # Projections are pure and content-addressed: another connection's already
        # hydrated parser cache can be reused. Missing texts still parse normally.
        if _LOADED_ANALYSIS_VERSION == ANALYSIS_VERSION and _ANALYSIS_PAYLOADS:
            self._analysis_loaded = ANALYSIS_VERSION
            return
        cursor = 0
        while True:
            rows = await self._database.fetchall(
                "SELECT rowid,kind,text,payload_json FROM language_exchange_analysis "
                "WHERE parser_version=? AND rowid>? ORDER BY rowid LIMIT 2048",
                (ANALYSIS_VERSION, cursor))
            if not rows:
                break
            for row in rows:
                _remember_analysis(row["kind"], row["text"], row["payload_json"])
            cursor = rows[-1]["rowid"]
            await asyncio.sleep(0)
        self._analysis_loaded = ANALYSIS_VERSION
        _LOADED_ANALYSIS_VERSION = ANALYSIS_VERSION

    async def _persist_analysis(self, prompt, answer):
        data = await asyncio.to_thread(self._analysis_entries, prompt, answer)
        async with self._database.transaction() as connection:
            await connection.executemany(
                "INSERT OR IGNORE INTO language_exchange_analysis VALUES (?,?,?,?,?)", data)

    @staticmethod
    def _analysis_entries(prompt, answer):
        """Pure cache construction must not monopolize the response loop."""
        entries = [(kind, text, function(text)) for text in (prompt, answer)
                   for kind, function in (("exchange", exchange_analysis),
                       ("reaction", reactive_act), ("features", turn_features),
                       ("units", lambda text: _question_units("", text))) ]
        entries.append(("forms", json.dumps([prompt, answer], ensure_ascii=False),
                        cached_response_forms(prompt, answer)))
        data = []
        for kind, text, value in entries:
            payload = json.dumps(_analysis_data(value), ensure_ascii=False, separators=(",", ":"))
            _remember_analysis(kind, text, payload)
            data.append((hashlib.sha256(text.encode()).hexdigest(), ANALYSIS_VERSION,
                         kind, text, payload))
        return data

    async def warm_analysis_cache(self, *, batch_size=16, max_exchanges=None):
        """Cooperative maintenance entry point, also callable by a future sleep cycle."""
        await self._load_analysis_cache()
        rows = await self._database.fetchall(
            "SELECT p.content prompt,a.content answer FROM language_observed_exchanges x "
            "JOIN language_discourse_turns p ON p.source_id=x.prompt_event_id "
            "JOIN language_discourse_turns a ON a.source_id=x.answer_event_id "
            "ORDER BY x.observed_at,x.answer_event_id")
        processed = 0
        for i, row in enumerate(rows):
            text = json.dumps([row["prompt"], row["answer"]], ensure_ascii=False)
            if ("forms", text) not in _ANALYSIS_PAYLOADS:
                if max_exchanges is not None and processed >= max_exchanges:
                    return False
                await self._persist_analysis(row["prompt"], row["answer"])
                processed += 1
            if i % batch_size == 0:
                await asyncio.sleep(0)


        # Prime the same reversible forms projection while the gateway starts.
        # Build a private projection in a worker; publish only if no live reader
        # has populated it meanwhile. No response waits for this warmup.
        guilds = await self._database.fetchall(
            "SELECT DISTINCT guild_id FROM language_observed_exchanges ORDER BY guild_id")
        for guild in guilds:
            identifier = guild[0]
            if self._exchange_rows.get(identifier):
                continue
            projected = await self._database.fetchall(
                "SELECT x.*,p.content prompt,a.content answer,p.actor_id prompt_actor_id "
                "FROM language_observed_exchanges x "
                "JOIN language_discourse_turns p ON p.source_id=x.prompt_event_id "
                "JOIN language_discourse_turns a ON a.source_id=x.answer_event_id "
                "WHERE x.guild_id=? ORDER BY x.observed_at DESC,x.answer_event_id LIMIT 8192",
                (identifier,))
            await self._warm_projection(identifier, projected)
        # Obsolete projections contain no evidence. Retire them cooperatively,
        # after publishing the usable projection; live readers keep their WAL view.
        while True:
            obsolete = await self._database.fetchall(
                "SELECT rowid FROM language_exchange_analysis WHERE parser_version!=? LIMIT 256",
                (ANALYSIS_VERSION,))
            if not obsolete:
                return True
            await self._database.execute(
                "DELETE FROM language_exchange_analysis WHERE rowid IN (" +
                ",".join("?" for _ in obsolete) + ")", tuple(r[0] for r in obsolete))
            await asyncio.sleep(0)
            if max_exchanges is not None:
                return False


    async def _warm_projection(self, guild_id, rows):
        task = self._projection_jobs.get(guild_id)
        if task is None or task.done():
            existing = self._exchange_rows.get(guild_id)
            if existing:
                return existing

            def build():
                projection = _ExchangeRows()
                projection.update(rows)
                return projection

            task = asyncio.create_task(asyncio.to_thread(build))
            self._projection_jobs[guild_id] = task
            self._database._maintenance_tasks.add(task)
            task.add_done_callback(self._database._maintenance_tasks.discard)
        projection = await asyncio.shield(task)
        if not self._exchange_rows.get(guild_id):
            self._exchange_rows[guild_id] = projection
        return self._exchange_rows[guild_id]

    async def _cached_rows(self, name, event, sql, parameters):
        # Cache bounded projections, never a response or a participant's fact.
        # Local observations/feedback invalidate them; time travel misses the cache.
        key = (name, event.guild_id)
        cached = self._read_cache.get(key)
        stamp = event.utc_timestamp.isoformat()
        watermark = await self._database.fetchone(
            "SELECT COUNT(*), MAX(rowid) FROM language_observed_exchanges "
            "WHERE guild_id=? AND observed_at < ?",
            (event.guild_id, stamp))
        revision = (*tuple(watermark), self._exchange_revision)
        if cached and cached[0] == revision and cached[1] <= stamp:
            return cached[2]
        rows = await self._database.fetchall(sql, parameters)
        if len(self._read_cache) >= 16:
            self._read_cache.clear()
        if name == "forms":
            projection = self._exchange_rows.get(event.guild_id)
            if not projection:
                projection = await self._warm_projection(event.guild_id, rows)
            projection.update(rows)
            rows = projection
        self._read_cache[key] = (revision, stamp, rows)
        return rows


    async def observed_operations(self, event) -> dict:
        """Lexical operations (greeting, «помнить») as people used them before this event."""
        # They come from human observations, never bot repetitions.
        rows = await self._database.fetchall(
            """SELECT d.source_id, d.content FROM language_discourse_turns d
               JOIN events e ON e.event_id = d.source_id WHERE d.guild_id = ?
                 AND d.observed_at < ?
                 AND (d.content LIKE '%приветств%' OR d.content LIKE '%помн%')
                 AND COALESCE(json_extract(e.payload_json, '$.author_is_bot'), 0) = 0
               ORDER BY d.observed_at DESC LIMIT 512""",
            (event.guild_id, event.utc_timestamp.isoformat()),
        )
        operations = {}
        from debilgi.language.clarification_policy import CONTACT_WORDS
        from debilgi.language.lexical_matching import same_lexical_family
        for source in rows:
            text = source["content"]
            if not ("приветств" in text.casefold() or "помн" in text.casefold()):
                continue
            for token in exchange_analysis(text)[0].tokens:
                if token.part_of_speech == "verb" and token.lemma == "помнить":
                    operations.setdefault("epistemic", (token.lemma, source["source_id"]))
                if (token.part_of_speech == "verb" and (
                    "greeting" in analyze_pragmatics(token.surface).functions
                    or any(same_lexical_family(token.lemma, w) for w in CONTACT_WORDS))):
                    operations.setdefault("greeting", (token.lemma, source["source_id"]))
        return operations

    async def prior_response_for(self, event: Event):
        """A delivered turn to this person. An explicit reply edge wins and has no time window;
        without one only the last ten minutes count."""
        key = ("receipt", event.event_id, event.guild_id, event.channel_id, event.actor_id,
               event.utc_timestamp, event.payload.get("reply_to_message_id"))
        if key in self._read_cache:
            return self._read_cache[key]
        parent = event.payload.get("reply_to_message_id")
        row = await self._database.fetchone(
            """SELECT i.prompt_event_id, i.response_message_id, i.response_text, i.plan_json,
                      i.selection_json, i.sent_at
               FROM language_meaning_intentions i
               WHERE i.guild_id = ? AND i.channel_id = ? AND i.user_id = ?
                 AND i.response_message_id IS NOT NULL AND i.sent_at < ?
                 AND (i.response_message_id = ? OR (? IS NULL AND i.sent_at >= ?))
               ORDER BY i.sent_at DESC LIMIT 1""",
            (event.guild_id, event.channel_id, event.actor_id, event.utc_timestamp.isoformat(),
             parent, parent, (event.utc_timestamp - timedelta(minutes=10)).isoformat()),
        )
        if not row:
            return None
        result = dict(row)
        result["plan"] = json.loads(result.pop("plan_json"))
        selection = json.loads(result.pop("selection_json"))
        result["decision"] = selection.get("response_decision", {})
        result["basis"] = selection.get("basis") or {}
        if not result["decision"]:
            trace = await self._database.fetchone(
                "SELECT s.details_json FROM language_response_traces t "
                "JOIN language_response_trace_steps s ON s.trace_id=t.trace_id "
                "WHERE t.source_event_id=? AND s.stage='intent' AND s.component='response_mode' "
                "ORDER BY s.sequence DESC LIMIT 1", (result["prompt_event_id"],))
            result["decision"] = json.loads(trace[0]).get("decision", {}) if trace else {}
        result["receipt_id"] = f"receipt:{event.channel_id}:{result['response_message_id']}"
        # A delivery event proves the speech act only. Its quoted hypothesis
        # is deliberately not inserted as a canonical world proposition.
        from debilgi.language.semantic_graph import spread_activation

        result["activation"] = spread_activation(
            {"delivered": 1.0}, (("delivered", result["receipt_id"], 1.0),
                                 (result["receipt_id"], "speech_act", 1.0)),
        ).get("speech_act", 0.0)
        result["graph_event"] = {
            "kind": "delivered_speech_act", "status": "Known",
            "evidence_id": result["receipt_id"], "speaker_id": event.payload.get("bot_user_id"),
            "addressee_id": event.actor_id, "channel_id": event.channel_id,
            "time": result["sent_at"], "quoted_plan": result["plan"],
        }
        operations = await self.observed_operations(event)
        result["operations"] = operations
        if len(self._read_cache) >= 32:
            self._read_cache.pop(next(iter(self._read_cache)))
        self._read_cache[key] = result
        return result

    async def explain_delivered_basis(self, event, context, prior):
        """«Why?» about the agent's own delivered turn: say the premise it really had.

        Only the agent's own record of the decision is read (``basis`` stored with the
        turn). A premise is admitted when it is a world fact of this server or something
        this same person said or did; corpus donors, other people's personal facts and
        grammar sources are never reasons. With no admissible premise the caller keeps
        the honest limit.
        """
        from debilgi.language.communication import MessagePlan, reader_goal
        from debilgi.language.dialogue_obligations import thought_from_data
        from debilgi.language.generative_planner import (
            DerivedPlan,
            SemanticEvidence,
            statement_thought,
        )
        from debilgi.language.productive_nlg import (
            ClauseThought,
            ConceptArgument,
            ConceptLexeme,
            SemanticDependency,
        )
        from debilgi.language.proposition_memory import world_thought

        basis = prior["basis"]
        premises = []
        for item in basis.get("premises", ()):
            ids = (item["evidence_id"], *item["thought"].get("evidence", ()))
            if item["kind"] not in {"canonical_claim", "learned_fact", "knowledge",
                                    "agent_observed"} or any(
                    str(i).startswith("seed:") for i in ids):
                continue
            thought = thought_from_data(item["thought"])
            if thought.clause_kind != "statement":
                continue
            if item["kind"] == "agent_observed":
                # The agent's own observed action, the ground of its supposition.
                row = await self._database.fetchone(
                    "SELECT 1 FROM meaning_claims WHERE claim_id=? AND guild_id=? "
                    "AND source_kind='agent_observed'", (item["evidence_id"], event.guild_id))
                if row is None:
                    continue
            elif item["kind"] == "canonical_claim":
                row = await self._database.fetchone(
                    "SELECT speaker_id,source_kind FROM meaning_claims WHERE claim_id=? "
                    "AND guild_id=?", (item["evidence_id"], event.guild_id))
                if row is None or row["source_kind"] == "corpus_form" or (
                        not world_thought(thought) and row["speaker_id"] != event.actor_id):
                    continue
            elif not world_thought(thought):
                continue
            premises.append(SemanticEvidence(item["evidence_id"], item["kind"], item["text"],
                                             thought))
        for event_id in (*basis.get("affect_events", ()), *basis.get("revised_because", ())):
            if str(event_id).startswith("seed:"):
                continue
            row = await self._database.fetchone(
                "SELECT d.content FROM affect_observations o JOIN language_discourse_turns d "
                "ON d.source_id=o.event_id WHERE o.event_id=? AND o.guild_id=? AND o.actor_id=?",
                (event_id, event.guild_id, event.actor_id))
            if row is None:
                continue
            text = row["content"]
            structure = exchange_analysis(text)[0]
            if any(c.clause_kind == "question" for c in structure.clauses):
                continue  # a question is not a reason
            greeting = prior["operations"].get("greeting")
            if "greeting" in analyze_pragmatics(text).functions and greeting:
                thought = ClauseThought(
                    ConceptLexeme(greeting[0], greeting[0], "verb", (
                        ("tense", "past"), ("number", "singular"), ("gender", "masculine"))),
                    (SemanticDependency("subject", ConceptArgument(
                        ConceptLexeme("ты", "ты", "pronoun"))),
                     SemanticDependency("direct_object", ConceptArgument(
                        ConceptLexeme("я", "я", "pronoun"), grammatical_case="accusative"))),
                    confidence=1.0, evidence=(event_id, greeting[1]))
            else:
                thought = statement_thought(text, speaker="user")
                if thought is None:
                    continue
                thought = replace(thought, evidence=(event_id,))
            premises.append(SemanticEvidence(event_id, "affect_event", text, thought))
        if not premises:
            return None
        premise = premises[0]
        plan = MessagePlan(reader_goal(context.structure), premise.thought,
                           mixture=(("known", 1.0),))
        return DerivedPlan(plan, "explain_delivered_basis", (premise,), "почему")

    async def verify_reported_position(self, event, context):
        """«You said that P»: check P against what the agent really delivered to this person.

        The reported meaning (key and sign) is compared with the positions of the agent's own
        delivered turns to the same person on the same server. Said (past: any turn, present:
        the latest position): the report is the turn itself with its receipt. Said the opposite:
        the actual turn is quoted, which is the factual objection. Nothing found: the honest
        limit (the observed «помнить» operation), never a denial, because the journal of
        delivered turns is not a proof of absence.
        """
        from debilgi.language.canonical_claims import ClaimQuery
        from debilgi.language.communication import KnowledgeLimit, MessagePlan, reader_goal
        from debilgi.language.dialogue_obligations import thought_from_data
        from debilgi.language.generative_planner import DerivedPlan
        from debilgi.language.productive_nlg import (
            ClauseThought,
            ConceptArgument,
            ConceptLexeme,
            SemanticDependency,
            unresolved_answer_thought,
        )
        from debilgi.language.proposition_memory import meaning_key
        from debilgi.language.semantic_graph import reported_proposition

        reported = reported_proposition(context.structure)
        if reported is None or reported.question or event.actor_id is None:
            return None
        history = await own_turn_history(
            self._database, event.guild_id, event.actor_id, event.utc_timestamp)
        key = meaning_key(reported.content)
        matches = []
        for item in history:
            try:
                if item["claim"] and meaning_key(thought_from_data(item["claim"])) == key:
                    matches.append(item)
            except (KeyError, TypeError, ValueError):
                continue
        if reported.tense != "past":
            matches = matches[:1]  # «ты говоришь»: the position held now
        sign = bool(reported.content.negated)
        said = [m for m in matches if bool(m["claim"].get("negated")) == sign]
        found = (said or matches or [None])[0]
        goal = reader_goal(context.structure)
        subject = SemanticDependency("subject", ConceptArgument(ConceptLexeme(
            "я", "я", "pronoun", (("number", "singular"), ("person", "first")))))
        if found is not None:
            return delivered_report_plan(goal, found)
        operation = (await self.observed_operations(event)).get("epistemic")
        question = ClauseThought(
            ConceptLexeme("сказать", "сказать", "verb", (
                ("tense", "past"), ("number", "singular"), ("gender", "masculine"))),
            (subject, SemanticDependency("direct_object", ConceptArgument(
                ConceptLexeme("что", "что", "pronoun")))),
            clause_kind="question", focus_role="direct_object", confidence=0.7)
        assertion = replace(unresolved_answer_thought(confidence=0.7), dependencies=(subject,))
        evidence = ()
        if operation:
            assertion = replace(assertion, predicate=replace(
                assertion.predicate, lemma=operation[0], surface=operation[0]))
            evidence = (operation[1],)
        query = ClaimQuery(event.guild_id, event.channel_id, event.actor_id,
                           event.utc_timestamp.isoformat(), event.event_id,
                           predicate="сказать", requested_role="direct_object")
        plan = MessagePlan(goal, replace(assertion, evidence=evidence),
                           knowledge_limit=KnowledgeLimit(query, question, "что"))
        return DerivedPlan(plan, "verify_reported_position", ())

    async def own_position_plans(self, event, context, planning) -> tuple:
        """Other plans of a position the agent already holds, instead of a limit about it.

        When the only plans for a question about the agent's own state are admissions of
        not knowing, and the agent did take a position on that very meaning with this person
        (however long ago), the competing plans are its own words (the report of the turn)
        and the ground it recorded for them. Which one is said is left to the arbiter, whose
        non-repetition already weighs the turns said before.
        """
        from debilgi.language.communication import reader_goal
        from debilgi.language.dialogue_obligations import thought_from_data
        from debilgi.language.proposition_memory import meaning_key
        from debilgi.language.semantic_graph import parse_proposition

        if (not planning.proposals or event.actor_id is None
                or any(p.plan.knowledge_limit is None for p in planning.proposals)):
            return ()
        asked = parse_proposition(context.structure, allow_question=True, perspective="user")
        if asked is None or not any(c.clause_kind == "question" for c in
                                    context.structure.clauses) and "?" not in context.prompt:
            return ()
        key = meaning_key(asked)
        held = []
        for item in await own_turn_history(
                self._database, event.guild_id, event.actor_id, event.utc_timestamp):
            try:
                if item["claim"] and meaning_key(thought_from_data(item["claim"])) == key:
                    held.append(item)
            except (KeyError, TypeError, ValueError):
                continue
        if not held:
            return ()
        plans = [delivered_report_plan(reader_goal(context.structure), held[0])]
        if held[0]["basis"]:
            prior = {"basis": held[0]["basis"], "operations": await self.observed_operations(event)}
            explained = await self.explain_delivered_basis(event, context, prior)
            if explained is not None:
                plans.append(explained)
        return tuple(plans)

    async def supposition_plans(self, event, context, planning) -> tuple:
        """A marked supposition instead of a bare limit, when the person asks for one.

        Licence comes from signals the reader already has: an explicit request to answer
        (``is_response_control``), a conditional frame (``advcl:condition``) or a question
        about the agent's own thinking. Without one the limit stays. The ground is only the
        agent's own record: its recent observed actions, or the questions it holds open for
        this person. Nothing is invented from other people's answers, and with no ground
        there is no plan (a limit is more honest than a coin toss). A supposition is never
        written to ``meaning_claims``.
        """
        from debilgi.config import thought_content_enabled
        from debilgi.language.communication import OWN_STATE_PROJECTIONS, reader_goal
        from debilgi.language.request_focus import is_response_control
        from debilgi.language.semantic_graph import conditional_proposition

        if (not planning.proposals or event.actor_id is None or event.channel_id is None
                or any(p.plan.knowledge_limit is None for p in planning.proposals)):
            return ()
        goal = reader_goal(context.structure)
        conditional = conditional_proposition(context.structure) is not None
        asked_state = OWN_STATE_PROJECTIONS.get(goal.predicate)
        if not (asked_state or conditional or is_response_control(event.meaning.text)
                or self._asks_own_doing(context)):
            return ()
        if asked_state == "open_questions":
            return await self._open_question_plans(event, context, goal)
        if thought_content_enabled() and self._asks_doing_on_a_day(context):
            # A day was named and the record has no deed of that day (else it would be Known):
            # a deed of another day, or of the last minutes, is no guess at what was asked.
            return ()
        if self._own_doing_query(context) and thought_content_enabled():
            if self._asks_frequency(context):
                # «Чем чаще всего?» is about the habit: a single recent doing is no answer to it.
                return await self._older_deed_plans(event, context, goal, (), habit_only=True)
            recent = await self._recent_action_plans(event, context, goal)
            return (*recent, *await self._older_deed_plans(event, context, goal, recent))
        return await self._recent_action_plans(event, context, goal)

    async def curiosity_plans(self, event, context, planning, *, irritation=None,
                              boredom=None) -> tuple:
        """A question about the unknown topic of a world question, next to the honest limit.

        Stage 4.38 (``DEBILGI_THOUGHT_CONTENT``). The topic is the subject of the question as the
        evaluative/definition reader sees it («Гитлер был прав?», «что такое шампунь?»). The plan is
        ``быть(тема, ?)`` with the question word chosen by the animacy of the topic token. Strength
        is ``U * I * (1 - B) * (1 - R)``: U falls with the people (and claims) that already speak of
        the topic, I is the curiosity drive times the trust in the asker, B is the boredom ledger
        reading for this very plan and topic, R the irritation. The plan carries the strength as
        its confidence and competes with the limit in ``meaning_intents`` by the drives curiosity
        and caution; nothing here is a threshold. Hard exclusions (strength 0): the topic is the
        agent or the asker's own, the turn is an attack, the topic was asked about before and
        answered. An answer is not taken here (stage 4.39).
        """
        from debilgi.config import thought_content_enabled

        if (not thought_content_enabled() or not planning.proposals or event.actor_id is None
                or event.channel_id is None
                or any(p.plan.knowledge_limit is None for p in planning.proposals)):
            return ()
        result = context.claim_result
        form = context.turn_form
        if (result is None or result.status != "Unknown" or form is None or not form.addressed
                or (irritation is not None and irritation.hostile)):
            return ()
        from debilgi.language.canonical_claims import self_query
        from debilgi.language.communication import MessagePlan, reader_goal
        from debilgi.language.generative_planner import DerivedPlan, SemanticEvidence
        from debilgi.language.semantic_graph import (
            ClauseThought,
            ConceptArgument,
            ConceptLexeme,
            SemanticDependency,
            parse_proposition,
        )

        if self_query(result.query):
            return ()  # the topic is the agent itself
        structure = context.structure
        if any(t.lemma in _DEIXIS for t in structure.tokens):
            return ()  # the agent or the asker's own matters, not a topic of the world
        asked = parse_proposition(structure, allow_question=True, perspective="user",
                                  evaluation=True, definition=True)
        if asked is None or asked.predicate.lemma != "быть":
            return ()
        subject = next((d.argument.head for d in asked.dependencies if d.role == "subject"), None)
        token = next((t for t in structure.tokens
                      if subject is not None and t.lemma == subject.lemma
                      and t.part_of_speech == "noun"), None)
        if token is None or not token.lemma:
            return ()
        if not token.morphology_known and DEFAULT_MORPHOLOGY.has_known_neighbour(token.surface):
            return ()  # a typo of a known word, not an unknown topic
        topic = token.lemma
        # U: the people (not the asker, not bots) and the claims that already speak of the topic.
        stem = topic[:max(3, len(topic) - 2)]
        rows = await self._database.fetchall(
            "SELECT d.actor_id,d.content FROM language_discourse_turns d "
            "JOIN events e ON e.event_id=d.source_id WHERE d.guild_id=? AND d.observed_at<? "
            "AND d.actor_id IS NOT NULL AND d.actor_id!=? "
            "AND COALESCE(json_extract(e.payload_json,'$.author_is_bot'),0)=0 "
            "AND (d.content LIKE ? OR d.content LIKE ? OR d.content LIKE ?) "
            "ORDER BY d.observed_at DESC LIMIT 512",
            (event.guild_id, event.utc_timestamp.isoformat(), event.actor_id,
             *(f"%{form}%" for form in _letter_cases(stem))))
        speakers = {r["actor_id"] for r in rows
                    if any(DEFAULT_MORPHOLOGY.lemma(w) == topic
                           for w in re.findall(r"\w+", (r["content"] or "").casefold()))}
        claims = await self._database.fetchone(
            "SELECT COUNT(DISTINCT source_event_id) n FROM meaning_claims WHERE guild_id=? "
            "AND source_kind!='agent_observed' AND source_time<? AND subject_ref LIKE ?",
            (event.guild_id, event.utc_timestamp.isoformat(), f"%{topic}%"))
        unknown = 1.0 / (1.0 + len(speakers) + (claims["n"] if claims else 0))
        # Already asked this person about this topic and got an answer: the next turn of theirs
        # after the question, within the feedback window, that is not the question again.
        asked_before = await self._database.fetchall(
            "SELECT sent_at,channel_id,plan_json FROM language_meaning_intentions WHERE guild_id=? "
            "AND user_id=? AND operation='curiosity' AND response_message_id IS NOT NULL "
            "AND sent_at<? ORDER BY sent_at DESC LIMIT 32",
            (event.guild_id, event.actor_id, event.utc_timestamp.isoformat()))
        for row in asked_before:
            if topic not in _plan_topics(row["plan_json"]):
                continue
            later = await self._database.fetchone(
                "SELECT d.content FROM language_discourse_turns d WHERE d.guild_id=? AND "
                "d.channel_id=? AND d.actor_id=? AND d.observed_at>? AND d.observed_at<? "
                "AND d.observed_at<? ORDER BY d.observed_at LIMIT 1",
                (event.guild_id, row["channel_id"], event.actor_id, row["sent_at"],
                 (datetime.fromisoformat(row["sent_at"]) + timedelta(minutes=10)).isoformat(),
                 event.utc_timestamp.isoformat()))
            if later is not None and topic not in {
                    DEFAULT_MORPHOLOGY.lemma(w)
                    for w in re.findall(r"\w+", (later["content"] or "").casefold())}:
                return ()  # asked and answered
        from debilgi.personality.genome import DEBILGI_GENOME

        traits = DEBILGI_GENOME.traits()
        drive = max(0.0, traits["curiosity"] - 0.5
                    + dict(context.affect_features).get("state_interest", 0.0) * 0.3)
        trust = context.relation.vector.trust if context.relation is not None else 0.5
        interest = drive * trust
        animate = "anim" in token.grammemes
        word = "кто" if animate else "что"
        goal = reader_goal(structure)
        such = await self._asked_with_such(event, word)
        masculine = "femn" not in token.grammemes and "neut" not in token.grammemes
        verb = ConceptLexeme("быть", "быть", "verb", (("tense", "present"),))
        if such and (word == "что" or masculine):
            # «кто такой X» / «что такое X»: a form two people of this server were seen to use.
            gender = "neut" if word == "что" else "masc"
            thought = ClauseThought(
                verb,
                (SemanticDependency("subject", ConceptArgument(
                    ConceptLexeme(word, word, "pronoun"), grammatical_case="nominative")),
                 SemanticDependency("predicate_complement", ConceptArgument(
                     ConceptLexeme("такой", "такой", "adjective", (("gender", gender),)),
                     grammatical_case="nominative")),
                 SemanticDependency("topic", ConceptArgument(
                     ConceptLexeme(topic, token.surface, "noun", token.grammatical_features),
                     grammatical_case="nominative"))),
                clause_kind="question", focus_role="subject", predicate_class="copular",
                evidence=(event.event_id,))
        else:
            thought = ClauseThought(
                verb,
                (SemanticDependency("subject", ConceptArgument(
                    ConceptLexeme(topic, token.surface, "noun", token.grammatical_features))),
                 SemanticDependency("predicate_complement", ConceptArgument(
                     ConceptLexeme(word, word, "pronoun"), grammatical_case="nominative"))),
                clause_kind="question", focus_role="predicate_complement",
                predicate_class="copular", evidence=(event.event_id,))
        # Asked to define the topic by the person: the not-knowing stays and the question is asked
        # back (what the asker knows of it); otherwise the question stands alone.
        definition = goal.kind == "answer_slot" and goal.question_word in {"кто", "что"}
        limit = planning.proposals[0].plan
        if definition:
            back = ClauseThought(
                ConceptLexeme("знать", "знать", "verb", (("tense", "present"),)),
                (SemanticDependency("subject", ConceptArgument(
                    ConceptLexeme("ты", "ты", "pronoun"))),),
                clause_kind="question", predicate_class="lexical", evidence=(event.event_id,))
            plan = replace(limit, follow_up=back, mixture=(("known", 0.5), ("curiosity", 0.5)))
        else:
            plan = MessagePlan(goal, thought, speech_act="question",
                               mixture=(("curiosity", 1.0),))
        derived = DerivedPlan(plan, "curiosity", (
            SemanticEvidence(event.event_id, "unknown_topic", topic, asked),), word)
        tired = 0.0
        if boredom is not None:
            from debilgi.experience.meaning_intents import decay_weight

            # B of this question about this topic (whatever its wording): the decayed count of the
            # agent's own earlier inquiries about it, as in every boredom reading.
            repeats = sum(
                decay_weight(row.age, boredom.half_life) * (1.0 if row.person else 0.5)
                for row in boredom.rows if row.operation == "curiosity" and topic in row.material)
            tired = 1.0 - math.exp(-boredom.sensitivity * repeats)
        reaction = min(1.0, max(0.0, irritation.value)) if irritation is not None else 0.0
        strength = unknown * interest * (1.0 - tired) * (1.0 - reaction)
        if strength <= 0.0:
            return ()
        derived = replace(derived, premises=(*derived.premises, SemanticEvidence(
            event.event_id, "curiosity_strength", f"{strength:.4f}", asked)))
        if definition:
            return (replace(derived, plan=replace(plan, follow_up=replace(
                plan.follow_up, confidence=strength))),)
        thought = replace(thought, confidence=strength)
        return (replace(derived, plan=replace(plan, claim=thought)),)

    async def _asked_with_such(self, event, word: str) -> bool:
        """Two different people of this server were seen to ask «кто такой» / «что такое»."""
        rows = await self._database.fetchall(
            "SELECT d.actor_id,d.content FROM language_discourse_turns d "
            "JOIN events e ON e.event_id=d.source_id WHERE d.guild_id=? AND d.observed_at<? "
            "AND d.actor_id IS NOT NULL "
            "AND COALESCE(json_extract(e.payload_json,'$.author_is_bot'),0)=0 "
            "AND (d.content LIKE ? OR d.content LIKE ? OR d.content LIKE ?) "
            "ORDER BY d.observed_at DESC LIMIT 512",
            (event.guild_id, event.utc_timestamp.isoformat(),
             *(f"%{form}%" for form in _letter_cases(f"{word} так"))))
        people = set()
        for row in rows:
            words = re.findall(r"\w+", (row["content"] or "").casefold())
            if any(a == word and DEFAULT_MORPHOLOGY.lemma(b) == "такой"
                   for a, b in zip(words, words[1:], strict=False)):
                people.add(row["actor_id"])
        return len(people) >= 2

    @staticmethod
    def _own_doing_query(context) -> bool:
        """The unanswered question is about what the agent does or did (any tense, no named day)."""
        from debilgi.language.canonical_claims import RELATIVE_DAYS, self_query

        result = context.claim_result
        return bool(
            result is not None and result.status == "Unknown" and self_query(result.query)
            and result.query.predicate_parent in {"делать", "заниматься"}
            and result.query.deictic not in RELATIVE_DAYS)

    @staticmethod
    def _asks_doing_on_a_day(context) -> bool:
        from debilgi.language.canonical_claims import RELATIVE_DAYS, self_query

        result = context.claim_result
        return bool(
            result is not None and result.status == "Unknown" and self_query(result.query)
            and result.query.predicate_parent in {"делать", "заниматься"}
            and result.query.deictic in RELATIVE_DAYS)

    @staticmethod
    def _asks_frequency(context) -> bool:
        from debilgi.language.canonical_claims import FREQUENCY_ADVERBS

        return any(FREQUENCY_ADVERBS & {t.lemma, t.surface.casefold()}
                   for t in context.structure.tokens)

    @staticmethod
    def _asks_own_doing(context) -> bool:
        """«Что делаешь?» in the present, unanswered by the record: the agent's own doing is asked.

        The licence is the question itself (stage 4.32, ``DEBILGI_THOUGHT_CONTENT``): what the
        agent observed itself doing lately is the ground, and the person need not first ask for a
        guess. A doing that overlaps the moment of the question is already a Known claim.
        """
        from debilgi.config import thought_content_enabled
        from debilgi.language.canonical_claims import self_query

        result = context.claim_result
        return bool(
            thought_content_enabled() and result is not None and result.status == "Unknown"
            and self_query(result.query) and result.query.tense == "present"
            and result.query.predicate_parent in {"делать", "заниматься"})

    async def learned_hedge(self, event) -> tuple[str, str] | None:
        """The epistemic operator the most different people used, with one of its sources."""
        from debilgi.language.communication import EPISTEMIC_OPERATORS

        words = sorted(EPISTEMIC_OPERATORS)
        rows = await self._database.fetchall(
            "SELECT d.source_id,d.actor_id,d.content FROM language_discourse_turns d "
            "JOIN events e ON e.event_id=d.source_id WHERE d.guild_id=? AND d.observed_at<? "
            "AND COALESCE(json_extract(e.payload_json,'$.author_is_bot'),0)=0 AND ("
            + " OR ".join("d.content LIKE ?" for _ in words) + ") "
            "ORDER BY d.observed_at DESC LIMIT 512",
            (event.guild_id, event.utc_timestamp.isoformat(), *(f"%{w}%" for w in words)))
        people: dict[str, set] = {}
        source: dict[str, str] = {}
        for row in rows:
            for token in exchange_analysis(row["content"])[0].tokens:
                if token.lemma in EPISTEMIC_OPERATORS:
                    people.setdefault(token.lemma, set()).add(row["actor_id"])
                    source.setdefault(token.lemma, row["source_id"])
        if not people:
            return None
        name = max(people, key=lambda w: (len(people[w]), w))
        return name, source[name]

    # ------------------------------------------------------------------ fantasy (4.23)
    async def learned_amusement_marker(self, event) -> tuple[str, str] | None:
        """The laughter cue the most different people of this server close a statement with.

        Like ``learned_hedge`` it is observed, not listed: a turn's closing cue is read by the
        pragmatic reader as amusement, a vote is one author, and the cue with the most authors
        wins (ties: the shorter, then alphabetical). Another server's turns are never read.
        """
        rows = await self._database.fetchall(
            "SELECT d.source_id,d.actor_id,d.content FROM language_discourse_turns d "
            "JOIN events e ON e.event_id=d.source_id WHERE d.guild_id=? AND d.observed_at<? "
            "AND COALESCE(json_extract(e.payload_json,'$.author_is_bot'),0)=0 "
            "AND d.question_status='not_question' ORDER BY d.observed_at DESC LIMIT 2048",
            (event.guild_id, event.utc_timestamp.isoformat()))
        people: dict[str, set] = {}
        source: dict[str, str] = {}
        for row in rows:
            content = (row["content"] or "").strip()
            analysis = analyze_pragmatics(content)
            closing = content.casefold().rstrip(" .!)…")
            for cue in analysis.cues:
                word = cue.surface.casefold()
                if (cue.function == "amusement" and closing.endswith(word)
                        and len(closing) > len(word)):
                    people.setdefault(word, set()).add(row["actor_id"])
                    source.setdefault(word, row["source_id"])
        if not people:
            return None
        name = max(people, key=lambda w: (len(people[w]), -len(w), w))
        return name, source[name]

    async def laughter_in_conversation(self, event, text: str) -> bool:
        """Does this person's current turn, or a turn of theirs in the last ten minutes, laugh?"""
        if "amusement" in analyze_pragmatics(text).functions:
            return True
        form = event.turn_form
        if form is not None and form.kind == "amusement":
            return True
        rows = await self._database.fetchall(
            "SELECT d.content FROM language_discourse_turns d JOIN events e "
            "ON e.event_id=d.source_id WHERE d.guild_id=? AND d.channel_id=? AND d.actor_id=? "
            "AND d.observed_at<? AND d.observed_at>=? "
            "AND COALESCE(json_extract(e.payload_json,'$.author_is_bot'),0)=0 "
            "ORDER BY d.observed_at DESC LIMIT 8",
            (event.guild_id, event.channel_id, event.actor_id, event.utc_timestamp.isoformat(),
             (event.utc_timestamp - timedelta(minutes=10)).isoformat()))
        return any("amusement" in analyze_pragmatics(r["content"] or "").functions for r in rows)

    async def play_barred(self, event, text: str) -> str | None:
        """Why the agent must not play now: the person's own tone is negative."""
        negative = {"sadness", "anger", "shock_or_frustration", "aversion", "regret", "annoyance",
                    "annoyance_or_surprise", "coarse_emphasis", "bittersweet_emotion",
                    "skepticism_or_disappointment"}
        texts = [text]
        rows = await self._database.fetchall(
            "SELECT d.content FROM language_discourse_turns d JOIN events e "
            "ON e.event_id=d.source_id WHERE d.guild_id=? AND d.channel_id=? AND d.actor_id=? "
            "AND d.observed_at<? AND d.observed_at>=? "
            "AND COALESCE(json_extract(e.payload_json,'$.author_is_bot'),0)=0 "
            "ORDER BY d.observed_at DESC LIMIT 3",
            (event.guild_id, event.channel_id, event.actor_id, event.utc_timestamp.isoformat(),
             (event.utc_timestamp - timedelta(minutes=10)).isoformat()))
        texts.extend(r["content"] or "" for r in rows)
        for item in texts:
            if analyze_pragmatics(item).functions & negative:
                return "negative_tone"
            valence = lexical_valence(item)
            if valence is not None and valence < 0:
                return "negative_valence"
        return None

    async def fantasy_budget_left(self, event) -> bool:
        """No play right after play: the last turn delivered to this person was not a fantasy."""
        row = await self._database.fetchone(
            "SELECT selection_json FROM language_meaning_intentions WHERE guild_id=? AND "
            "channel_id=? AND user_id=? AND sent_at<? AND sent_at>=? AND "
            "response_message_id IS NOT NULL ORDER BY sent_at DESC LIMIT 1",
            (event.guild_id, event.channel_id, event.actor_id, event.utc_timestamp.isoformat(),
             (event.utc_timestamp - timedelta(minutes=10)).isoformat()))
        return row is None or not (json.loads(row["selection_json"]).get("basis") or {}).get(
            "fantasy")

    async def chat_theme(self, event, *, quiet: bool = False) -> dict[str, set]:
        """What the channel talks about now: lemma → the distinct people who used it (≤20 min).

        One person is one vote per lemma, so a single spammer's repetitions weigh as one.
        """
        from debilgi.language.content import content_lemmas

        sql = (
            "SELECT d.actor_id,d.content FROM language_discourse_turns d JOIN events e "
            "ON e.event_id=d.source_id WHERE d.guild_id=? AND d.channel_id=? AND d.observed_at<? "
            "AND {} d.source_id!=? "
            "AND COALESCE(json_extract(e.payload_json,'$.author_is_bot'),0)=0 "
            "ORDER BY d.observed_at DESC LIMIT 40")
        base = (event.guild_id, event.channel_id, event.utc_timestamp.isoformat())
        rows = () if quiet else await self._database.fetchall(
            sql.format("d.observed_at>=? AND"),
            (*base, (event.utc_timestamp - timedelta(minutes=20)).isoformat(), event.event_id))
        if not rows:  # a quiet channel: what it last talked about
            rows = await self._database.fetchall(sql.format(""), (*base, event.event_id))
        theme: dict[str, set] = {}
        for row in rows:
            for lemma in content_lemmas(row["content"] or ""):
                theme.setdefault(lemma, set()).add(row["actor_id"])
        return theme

    async def fantasy_material(self, event, context, lemma, *, role, avoid=(), nested=False):
        """A noun people of this server use with this verb, as an argument of the asked role.

        Associations come from learned word relations (no participants): the governor is the
        verb's family (any aspect, prefix or determinacy), the dependent a common noun
        of non-negative tone, seen ≥2 times and — read from the corpus itself — with at least
        two different authors. Among those the one the chat is talking about now (distinct
        recent people) is preferred, then the most frequent. Returns
        ``(argument, evidence_id)`` or ``None``. ``nested`` asks for the verb an infinitive stands
        for («хотел бы быть …»: material for «быть», not for «хотеть»); a short stem has no
        prefixed family then («забыть» is not «быть»), only the verb itself counts.
        """
        from debilgi.language.lexical_matching import verb_family
        from debilgi.language.productive_nlg import ConceptArgument, ConceptLexeme
        from debilgi.language.word_relations import observe_word_relations

        wanted = "obj" if role == "direct_object" else "obl"

        def same_verb(governor: str) -> bool:
            if nested and len(lemma) <= 4:
                return governor == lemma
            return verb_family(governor, lemma)

        candidates = {}
        # How much of a noun's use is with this verb: a hub noun used with every verb
        # («время», «работа») says nothing about this one.
        total: dict[str, int] = {}
        for relation in context.word_relations or ():
            if relation.dependent_pos == "noun" and relation.relation.partition(":")[0] in {
                    "obj", "obl"}:
                total[relation.dependent] = total.get(relation.dependent, 0) + relation.count
        for relation in context.word_relations or ():
            if (relation.dependent_pos != "noun" or relation.count < 2
                    or (relation.relation != "obj" if wanted == "obj"
                        else relation.relation != "obl:location_or_destination")
                    or relation.dependent in avoid
                    or not same_verb(relation.governor)):
                continue
            candidates.setdefault((relation.dependent, relation.case), 0)
            candidates[(relation.dependent, relation.case)] += relation.count
        theme = await self.chat_theme(event)
        specific = {key for key in candidates
                    if candidates[key] / max(1, total.get(key[0], 0)) >= 0.08}
        ranked = sorted(specific, key=lambda k: (-len(theme.get(k[0], ())), -candidates[k], k))
        for dependent, case in ranked[:6]:
            if not self._fit_for_play(dependent):
                continue
            preposition = None
            if wanted == "obl":
                options = sorted(
                    ((r.dependent, r.count) for r in context.word_relations
                     if r.governor == dependent and r.relation == "case"
                     and case in DEFAULT_MORPHOLOGY.expected_cases(r.dependent)),
                    key=lambda item: (-item[1], item[0]))
                if not options:
                    continue
                preposition = options[0][0]
            # Two different authors must have said it (a spammer's repeats are one voice).
            stem = dependent[:max(3, len(dependent) - 2)]
            turns = await self._database.fetchall(
                "SELECT d.actor_id,d.content FROM language_discourse_turns d JOIN events e "
                "ON e.event_id=d.source_id WHERE d.guild_id=? AND d.observed_at<? AND "
                "d.content LIKE ? AND COALESCE(json_extract(e.payload_json,'$.author_is_bot'),0)=0 "
                "ORDER BY d.observed_at DESC LIMIT 160",
                (event.guild_id, event.utc_timestamp.isoformat(), f"%{stem}%"))
            authors = {
                t["actor_id"] for t in turns
                if any(r.dependent == dependent and same_verb(r.governor)
                       for r in observe_word_relations(t["content"] or "",
                                                       human_perspective=True))}
            if len(authors) < 2:
                continue
            # Nominative is the case of a subject or of a copular clause; an object of an
            # infinitive («хотел бы быть …») never has it, the embedded verb governs instead.
            if nested and case == "nominative":
                case = ""
            argument = ConceptArgument(
                ConceptLexeme(dependent, dependent, "noun", (("number", "singular"),)),
                preposition=preposition, grammatical_case=case or None)
            return argument, f"word_relation:{lemma}:{dependent}:{wanted}"
        return None

    @staticmethod
    def _fit_for_play(lemma: str) -> bool:
        """A common noun of non-negative tone: the only kind a game is made of (no names, no
        words of sadness or anger). The same screen for corpus material and for a complement
        borrowed from an earlier question."""
        negative = {"sadness", "anger", "shock_or_frustration", "aversion", "coarse_emphasis"}
        reading = DEFAULT_MORPHOLOGY.best(lemma)
        if (reading is None or not reading.is_known
                or {"Name", "Surn", "Geox", "Orgn", "Patr"} & set(reading.grammemes)):
            return False
        valence = lexical_valence(lemma)
        return not ((valence is not None and valence < 0)
                    or analyze_pragmatics(lemma).functions & negative)

    @staticmethod
    def _earlier_human_texts(context, current: str, limit: int = 6) -> tuple[str, ...]:
        """People's earlier turns of this conversation, latest first (never the agent's own
        words, never the current turn)."""
        texts = []
        for message in reversed(tuple(getattr(context, "consultation_messages", ()) or ())):
            speaker, separator, text = message.partition(": ")
            if separator and speaker.casefold() == "user" and text.strip() != current.strip():
                texts.append(text)
        return tuple(texts[:limit])

    def _own_frame(self, event, context):
        """What was asked of the agent, as a statement about itself: ``(thought, slot_role,
        irrealis)`` or ``None``. A «бы» question, or «если C, то P бы», retold in the agent's
        person; a wh-slot is dropped (the answer fills it)."""
        from debilgi.language.communication import reader_goal
        from debilgi.language.productive_nlg import ConceptArgument, ConceptLexeme
        from debilgi.language.semantic_graph import (
            complete_nested_complement,
            conditional_proposition,
            parse_proposition,
        )

        structure = context.structure
        goal = reader_goal(structure)
        asked = ConceptArgument(ConceptLexeme("я", "я", "pronoun"))
        conditional = conditional_proposition(structure, possessive_deixis=True)
        if conditional is not None and conditional.irrealis:
            thought, irrealis = conditional.consequent, True
        else:
            thought = parse_proposition(
                structure, allow_question=True, fallback_subject=asked,
                possessive_deixis=True, irrealis=True, tree=True)
            if thought is not None:
                thought = complete_nested_complement(
                    thought, self._earlier_human_texts(context, structure.text))
            irrealis = bool(thought is not None
                            and dict(thought.predicate.features).get("mood") == "conditional")
        if thought is None or not any(
                d.role == "subject" and d.argument.head.lemma == "я" for d in thought.dependencies):
            return None
        slot = {"что": "direct_object", "куда": "location", "где": "location",
                "откуда": "location"}.get(goal.question_word)
        if goal.question_word and slot is None:
            return None  # a slot the relations cannot fill (кто, как, когда, сколько …)
        thought = replace(thought, dependencies=tuple(
            d for d in thought.dependencies if d.argument.head.lemma != goal.question_word))
        return thought, slot, irrealis

    async def fantasy_plans(self, event, context, planning, *, continuation=None,
                            said=frozenset()) -> tuple:
        """Make-believe about the agent itself, marked as play, when play is licensed.

        Licence (PLAN_422_fantasy §3.2, decisions 8–9): a question in the unreal mood about
        the agent (F1); laughter in the conversation with a question about the agent (F2);
        a request to make something up (F3). Never: a grounded answer or a guess exists, a
        reason is asked, the subject is not the agent, the person's tone is negative, the
        agent has no playfulness, or the last turn was already play. A fantasy carries
        ``kind="fantasy"`` (the grammar of the unreal and/or the learned laughter marker),
        is never a fact of anyone and is never written to claims.
        """

        from debilgi.language.communication import MessagePlan, reader_goal
        from debilgi.language.generative_planner import DerivedPlan, SemanticEvidence
        from debilgi.language.productive_nlg import (
            ClauseThought,
            SemanticDependency,
        )
        from debilgi.language.request_focus import is_invention_request
        from debilgi.personality.genome import DEBILGI_GENOME

        if event.actor_id is None or event.channel_id is None:
            return ()
        if any(p.plan.knowledge_limit is None for p in planning.proposals):
            return ()  # Known and a grounded guess outrank play
        goal = reader_goal(context.structure)
        if goal.question_word in {"почему", "зачем"} and continuation is None:
            return ()  # a reason is asked, not a game (but a game under way may go on)
        turn_text = event.meaning.text
        invited = is_invention_request(turn_text)
        frame = None if continuation else self._own_frame(event, context)
        if invited and not (frame and frame[2]):
            frame = None  # a request to make something up asks for no particular frame
        if continuation is not None:
            frame = (continuation, None, True)
        if frame is not None and not any(
                d.role == "subject" and d.argument.head.lemma == "я"
                for d in frame[0].dependencies):
            return ()  # never about the interlocutor or a third person
        irrealis = bool(frame and frame[2])
        if continuation is None and not (irrealis or invited) and frame is None:
            return ()  # nothing licenses play: no question about itself, no request, no unreal
        laughing = await self.laughter_in_conversation(event, turn_text)
        if continuation is None and not (irrealis or invited or laughing):
            return ()
        traits = {**DEBILGI_GENOME.traits(),
                  **(asdict(context.personality) if context.personality else {})}
        if traits.get("playfulness", traits["chaos"]) - 0.5 <= 0:
            return ()  # accepting the game is a trait of character, not a rule
        if await self.play_barred(event, turn_text):
            return ()
        if continuation is None and not await self.fantasy_budget_left(event):
            return ()
        marker = await self.learned_amusement_marker(event) if (laughing or invited
                                                               or continuation) else None
        conditional = irrealis or marker is None
        if frame is None:
            frame = await self._theme_frame(event, context)
            if frame is None:
                return ()
        thought, slot, _ = frame
        if "elliptical_complement" in thought.evidence:
            # What an earlier question attached to the embedded verb is only the person's
            # question; as the agent's own play it must pass the screen material passes.
            thought = replace(thought, dependencies=tuple(
                d for d in thought.dependencies
                if d.head is None or self._fit_for_play(d.argument.head.lemma)))
        # «хотел бы быть …»: what is attached is attached to the embedded verb, so is the
        # material the play looks for (the modal governs only its infinitive).
        nested = (next((d for d in thought.dependencies if d.role == "action_complement"), None)
                  if thought.predicate_class == "modal" else None)
        head_role = nested.role if nested is not None else None
        lemma = nested.argument.head.lemma if nested is not None else thought.predicate.lemma
        avoid = {d.argument.head.lemma for d in thought.dependencies}
        # Never the same material twice in a row of play (ten minutes).
        avoid |= set(said) | await self._played_lemmas(event)
        # What the asked slot wants, else whatever the verb is seen to take: an object first,
        # then a place.
        roles = (slot,) if slot else tuple(
            r for r in ("direct_object", "location")
            if not any(d.role == r and d.head == head_role for d in thought.dependencies))
        material, role = None, roles[0] if roles else "location"
        for candidate_role in roles:
            material = await self.fantasy_material(
                event, context, lemma, role=candidate_role, avoid=avoid,
                nested=nested is not None)
            if material is not None:
                role = candidate_role
                break
        if material is None and (slot is not None or continuation is not None or not irrealis):
            # A slot cannot stay empty, a game goes on only with something new, and a joke told
            # in the indicative (laughter without «бы») needs substance of its own.
            return ()
        features = tuple(f for f in thought.predicate.features if f[0] not in {"mood", "tense"})
        features += (("tense", "past"),) + ((("mood", "conditional"),) if conditional else ())
        dependencies = thought.dependencies
        evidence = [marker[1]] if marker else []
        premises = [SemanticEvidence(event.event_id, "fantasy_frame", turn_text, thought)]
        if material is not None:
            argument, material_id = material
            dependencies = (*dependencies, SemanticDependency(role, argument, head=head_role))
            evidence.append(material_id)
        claim = ClauseThought(
            replace(thought.predicate, features=features), dependencies, clause_kind="statement",
            negated=thought.negated,
            # As sure of what it plays as of having understood the game it was offered.
            confidence=max(thought.confidence, 0.7), evidence=tuple(evidence),
            predicate_class=thought.predicate_class)
        if material is not None:
            premises.append(SemanticEvidence(material_id, "fantasy_material", "", claim))
        hypothesis = (("kind", "fantasy"),) + (
            (("operator", marker[0]), ("operator_source", marker[1])) if marker else ())
        plan = MessagePlan(goal, claim, speech_act="opinion", hypothesis=hypothesis,
                           mixture=(("fantasy", 1.0),))
        return (DerivedPlan(plan, "supposition", tuple(premises), goal.question_word),)

    async def _played_lemmas(self, event) -> set:
        """What the agent has already used in its play with this person (last ten minutes)."""
        rows = await self._database.fetchall(
            "SELECT plan_json,selection_json FROM language_meaning_intentions WHERE guild_id=? "
            "AND channel_id=? AND user_id=? AND response_message_id IS NOT NULL AND sent_at<? "
            "AND sent_at>=? ORDER BY sent_at DESC LIMIT 12",
            (event.guild_id, event.channel_id, event.actor_id, event.utc_timestamp.isoformat(),
             (event.utc_timestamp - timedelta(minutes=10)).isoformat()))
        used = set()
        for row in rows:
            if (json.loads(row["selection_json"]).get("basis") or {}).get("fantasy"):
                claim = json.loads(row["plan_json"]).get("claim") or {}
                used.update(d["argument"]["head"]["lemma"] for d in claim.get("dependencies", ()))
        return used

    async def _theme_frame(self, event, context):
        """A predicate for a free request to make something up: the verb of the chat's theme
        that has material (a noun to do it with), preferring what more distinct people said."""
        from debilgi.language.productive_nlg import (
            ClauseThought,
            ConceptArgument,
            ConceptLexeme,
            SemanticDependency,
        )

        verbs = {}
        for relation in context.word_relations or ():
            if (relation.relation == "obj" or relation.relation == "obl:location_or_destination"
                    ) and relation.count >= 2:
                verbs.setdefault(relation.governor, 0)
        for quiet in (False, True):  # what is talked about now, else what was talked about
            theme = await self.chat_theme(event, quiet=quiet)
            scored = []
            for governor in verbs:
                people = set().union(*(theme.get(g, set()) for g in theme
                                       if g == governor or _same_verb(g, governor)))
                if people:
                    scored.append((len(people), governor))
            for _, governor in sorted(scored, key=lambda item: (-item[0], item[1])):
                reading = DEFAULT_MORPHOLOGY.best(governor)
                if reading is None or reading.part_of_speech != "verb":
                    continue
                thought = ClauseThought(
                    ConceptLexeme(governor, governor, "verb", (
                        ("tense", "past"), ("gender", "masculine"), ("number", "singular"))),
                    (SemanticDependency("subject", ConceptArgument(ConceptLexeme(
                        "я", "я", "pronoun", (("number", "singular"), ("person", "first"))))),),
                    confidence=0.5)
                for role in ("location", "direct_object"):
                    if await self.fantasy_material(
                            event, context, governor, role=role) is not None:
                        return thought, role, False
        return None

    async def fantasy_followup_plans(self, event, context) -> tuple:
        """«Почему?» after the agent's own play: own up to it, or go on with the game.

        Both are plans of the same turn; which one is said is left to the existing choice
        between meanings (experience, state, repetition) and the arbiter — nothing is drawn
        from a list and no branch is fixed.
        """
        from debilgi.language.communication import (
            OWN_OPERATION_VERBS,
            MessagePlan,
            reader_goal,
        )
        from debilgi.language.dialogue_obligations import thought_from_data
        from debilgi.language.generative_planner import DerivedPlan, SemanticEvidence
        from debilgi.language.productive_nlg import (
            ClauseThought,
            ConceptArgument,
            ConceptLexeme,
            SemanticDependency,
        )

        variable = ellipsis_variable(context.prompt)
        prior = context.prior_bot_turn
        if (variable is None or variable["role"] != "cause" or not prior
                or not (prior.get("basis") or {}).get("fantasy")):
            return ()
        goal = reader_goal(context.structure)
        receipt = prior["receipt_id"]
        verb = OWN_OPERATION_VERBS["fantasy"]
        admission = ClauseThought(
            ConceptLexeme(verb, verb, "verb", (
                ("tense", "past"), ("gender", "masculine"), ("number", "singular"))),
            (SemanticDependency("subject", ConceptArgument(ConceptLexeme(
                "я", "я", "pronoun", (("number", "singular"), ("person", "first"))))),),
            confidence=0.9, evidence=(receipt,))
        plans = [DerivedPlan(
            MessagePlan(goal, admission, speech_act="admission", mixture=(("known", 1.0),)),
            "explain_delivered_basis",
            (SemanticEvidence(receipt, "sent_turn", prior["response_text"], admission),),
            "почему")]
        # Going on with the game: the same act, another thing to do it with.
        said = thought_from_data(prior["plan"]["claim"])
        # What the game does stays (a modal keeps its infinitive: «хотел бы быть …» goes on as
        # «хотел бы быть <another thing>»); what it did it with is dropped.
        keep = {"subject", "action_complement"} if said.predicate_class == "modal" else {"subject"}
        bare = replace(said, dependencies=tuple(
            d for d in said.dependencies if d.role in keep and d.head is None), evidence=())
        from debilgi.language.generative_planner import PlanningResult

        more = await self.fantasy_plans(
            event, context, PlanningResult(None, (), (), ()),
            continuation=replace(bare, predicate=said.predicate),
            said={d.argument.head.lemma for d in said.dependencies})
        more = tuple(replace(m, plan=replace(m.plan, hypothesis=(
            *m.plan.hypothesis, ("continues", "play")))) for m in more)
        return (*plans, *more)

    async def elided_limit_observed(self, event) -> bool:
        """Do ≥2 different people of this server say «не знаю» with the person carried by the
        verb (an observed ``express_limit`` reaction with no subject word)?"""
        rows = await self._database.fetchall(
            "SELECT x.response_forms_json,d.actor_id FROM language_observed_exchanges x "
            "JOIN language_discourse_turns d ON d.source_id=x.answer_event_id "
            "WHERE x.guild_id=? AND x.observed_at<? AND x.response_forms_json LIKE ? "
            "LIMIT 256",
            (event.guild_id, event.utc_timestamp.isoformat(), "%express_limit%"))
        people = set()
        for row in rows:
            for form in json.loads(row["response_forms_json"]):
                if (form.get("meaning") == "express_limit" and form.get("component") == "core"
                        and not any(s.get("lemma") in {"я", "мы"} for s in form.get("slots", ()))):
                    people.add(row["actor_id"])
        return len(people) >= 2

    async def is_human_utterance(self, event, text: str) -> bool:
        """Is this wording (marker aside) something a person of this server already said?"""
        bare = text.rstrip(" .!").rsplit(" ", 1)[0] if " " in text else text
        wanted = {normalized_response(text), normalized_response(bare)}
        probe = next((w for w in reversed(bare.casefold().split()) if len(w) >= 4), bare)
        rows = await self._database.fetchall(
            "SELECT d.content FROM language_discourse_turns d JOIN events e "
            "ON e.event_id=d.source_id WHERE d.guild_id=? AND d.content LIKE ? AND "
            "COALESCE(json_extract(e.payload_json,'$.author_is_bot'),0)=0 LIMIT 200",
            (event.guild_id, f"%{probe}%"))
        return any(normalized_response(r["content"] or "") in wanted for r in rows)

    async def _recent_action_plans(self, event, context, goal) -> tuple:
        """A guess about what the agent does now, from what it observed itself doing lately."""
        from debilgi.language.canonical_claims import answer_thought, self_query
        from debilgi.language.communication import MessagePlan
        from debilgi.language.generative_planner import DerivedPlan, SemanticEvidence
        from debilgi.language.proposition_memory import PropositionMemory

        result = context.claim_result
        if result is None or result.status != "Unknown" or not self_query(result.query):
            return ()
        hedge = await self.learned_hedge(event)
        if hedge is None:
            return ()  # a supposition must carry an observed epistemic marker
        # The same question, only without the strict tense and interval filter.
        query = replace(result.query, tense=None, valid_from=None, valid_to=None, deictic=None)
        found = await PropositionMemory.for_meaning(self._database).query_claims(
            query, reconcile=False)
        horizon = (event.utc_timestamp - timedelta(minutes=10)).isoformat()
        claims = sorted((c for c in found.claims
                         if c.source_kind == "agent_observed" and c.source_time >= horizon),
                        key=lambda c: c.source_time, reverse=True)
        plans, seen = [], set()
        for claim in claims:
            if claim.predicate in seen:
                continue
            seen.add(claim.predicate)
            observed = answer_thought(claim, query)
            thought = replace(observed, evidence=(*observed.evidence, hedge[1]))
            plan = MessagePlan(goal, thought, speech_act="opinion", hypothesis=(
                ("kind", "own_evidence_opinion"), ("operator", hedge[0]),
                ("operator_source", hedge[1])), mixture=(("hypothesis", 1.0),))
            plans.append(DerivedPlan(plan, "supposition", (
                SemanticEvidence(claim.claim_id, "agent_observed", "", observed),),
                goal.question_word))
            if len(plans) == 2:
                break
        return tuple(plans)

    async def learned_time_words(self, event) -> dict[str, int]:
        """Calendar, recency and frequency words with the number of different people who said them.

        Only invariable grammar words are looked for; a word counts for a person once, whatever
        the number of messages. The reading is this server's, up to this event.
        """
        from debilgi.language.canonical_claims import (
            FREQUENCY_ADVERBS,
            RECENCY_ADVERB,
            RELATIVE_DAYS,
        )

        words = sorted(
            {*RELATIVE_DAYS, RECENCY_ADVERB, *FREQUENCY_ADVERBS} - {"завтра", "послезавтра"})
        rows = await self._database.fetchall(
            "SELECT d.actor_id,d.content FROM language_discourse_turns d "
            "JOIN events e ON e.event_id=d.source_id WHERE d.guild_id=? AND d.observed_at<? "
            "AND COALESCE(json_extract(e.payload_json,'$.author_is_bot'),0)=0 AND ("
            + " OR ".join("d.content LIKE ?" for _ in words) + ") "
            "ORDER BY d.observed_at DESC LIMIT 2048",
            (event.guild_id, event.utc_timestamp.isoformat(), *(f"%{w}%" for w in words)))
        people: dict[str, set] = {}
        for row in rows:
            for found in re.findall(r"\w+", (row["content"] or "").casefold().replace("ё", "е")):
                if found in words:
                    people.setdefault(found, set()).add(row["actor_id"])
        return {w: len(a) for w, a in people.items()}

    async def _imperfective_partner(self, event, lemma: str) -> str | None:
        """The imperfective partner of a perfective own-doing verb, among verbs this server uses."""
        from debilgi.language.lexical_matching import aspect_pair

        rows = await self._database.fetchall(
            "SELECT DISTINCT governor FROM language_generation_word_relations WHERE guild_id=?",
            (event.guild_id,))
        near = sorted(r["governor"] for r in rows
                      if r["governor"] and r["governor"] != lemma
                      and (r["governor"][:3] == lemma[:3] or lemma.endswith(r["governor"][1:])))
        for word in near:
            if aspect_pair(word, lemma) and _same_root_pair(word, lemma):
                return word
        return None

    async def _older_deed_plans(self, event, context, goal, recent, habit_only=False) -> tuple:
        """What the agent did before the last ten minutes: a marked guess with an honest time.

        Stage 4.33 (``DEBILGI_THOUGHT_CONTENT``). A deed weighs 2^(-age/T) with T one day (the grain
        of the calendar words «сегодня», «вчера»; a deed whose weight fell below 1/8 is no longer
        «недавно»). Equal deeds are one material whose weight is the sum: that mass ranks the
        plans, the time word is that of the latest occurrence. A frequency question («чем
        чаще всего занимаешься?») gets the habit form: an imperfective verb the people of this
        server use, with a frequency adverb they use. No preference is read from it («нравится» is
        not derived). Nothing is invented: no hedge, no time word people use, no deed -- no plan.
        """
        from datetime import datetime
        from zoneinfo import ZoneInfo

        from debilgi.language.canonical_claims import (
            FREQUENCY_ADVERBS,
            RECENCY_ADVERB,
            RELATIVE_DAYS,
            answer_thought,
            self_query,
        )
        from debilgi.language.communication import MessagePlan
        from debilgi.language.generative_planner import DerivedPlan, SemanticEvidence
        from debilgi.language.productive_nlg import (
            ConceptArgument,
            ConceptLexeme,
            SemanticDependency,
        )
        from debilgi.language.proposition_memory import PropositionMemory

        result = context.claim_result
        if result is None or not self_query(result.query):
            return ()
        hedge = await self.learned_hedge(event)
        if hedge is None:
            return ()
        said = await self.learned_time_words(event)
        query = replace(result.query, tense=None, valid_from=None, valid_to=None, deictic=None)
        found = await PropositionMemory.for_meaning(self._database).query_claims(
            query, reconcile=False)
        now = event.utc_timestamp
        zone = ZoneInfo(query.calendar_zone) if query.calendar_zone else None
        horizon = (now - timedelta(minutes=10)).isoformat()
        taken = {p.plan.claim.predicate.lemma for p in recent}
        half_life = 86400.0
        groups: dict[tuple, dict] = {}
        for claim in found.claims:
            if claim.source_kind != "agent_observed" or claim.source_time >= horizon:
                continue
            age = (now - datetime.fromisoformat(claim.source_time)).total_seconds()
            weight = 2.0 ** (-age / half_life)
            if weight < 0.125:
                continue
            target = next((d.argument.head.lemma for d in claim.thought.dependencies
                           if d.role == "direct_object"), None)
            group = groups.setdefault(
                (claim.predicate, target), {"mass": 0.0, "count": 0, "last": claim})
            group["mass"] += weight
            group["count"] += 1
            if claim.source_time > group["last"].source_time:
                group["last"] = claim

        def day_word(claim) -> str | None:
            offset = (now.astimezone(zone).date()
                      - datetime.fromisoformat(claim.source_time).astimezone(zone).date()).days
            word = next((w for w, d in RELATIVE_DAYS.items() if d == -offset), None)
            if word in said and said[word] >= 2:
                return word
            return RECENCY_ADVERB if said.get(RECENCY_ADVERB, 0) >= 2 and offset >= 1 else None

        def adverbial(word):
            return SemanticDependency(
                "manner", ConceptArgument(ConceptLexeme(word, word, "adverb")))

        def plan_of(thought, claim, premise):
            plan = MessagePlan(goal, replace(thought, evidence=(*thought.evidence, hedge[1])),
                               speech_act="opinion", hypothesis=(
                ("kind", "own_evidence_opinion"), ("operator", hedge[0]),
                ("operator_source", hedge[1])), mixture=(("hypothesis", 1.0),))
            return DerivedPlan(plan, "supposition", (
                SemanticEvidence(claim.claim_id, "agent_observed", "", premise),),
                goal.question_word)

        plans = []
        ranked = sorted(groups.items(), key=lambda kv: (-kv[1]["mass"], kv[0][0], kv[0][1] or ""))
        if self._asks_frequency(context):
            frequent = max((w for w in FREQUENCY_ADVERBS if said.get(w, 0) >= 2 and w != "чаще"),
                           key=lambda w: (said[w], w), default=None)
            for (predicate, _), group in (ranked[:4] if frequent else ()):
                partner = await self._imperfective_partner(event, predicate)
                if partner is None:
                    continue
                claim = group["last"]
                observed = answer_thought(claim, query)
                habit = replace(
                    observed,
                    predicate=replace(observed.predicate, lemma=partner, surface=partner),
                    dependencies=(*observed.dependencies, adverbial(frequent)))
                plans.append(plan_of(habit, claim, observed))
                break
        for (predicate, _), group in ([] if habit_only else ranked):
            if predicate in taken or len(plans) >= 3:
                continue
            claim = group["last"]
            word = day_word(claim)
            if word is None:
                continue
            observed = answer_thought(claim, query)
            bound = replace(observed, dependencies=(*observed.dependencies, adverbial(word)))
            plans.append(plan_of(bound, claim, bound))
        return tuple(plans)

    async def _open_question_plans(self, event, context, goal) -> tuple:
        """What the agent is thinking about: the questions left open for this person."""
        from debilgi.language.canonical_claims import ClaimQuery
        from debilgi.language.communication import (
            OWN_STATE_PROJECTIONS,
            KnowledgeLimit,
            MessagePlan,
        )
        from debilgi.language.generative_planner import DerivedPlan, SemanticEvidence
        from debilgi.language.productive_nlg import (
            ClauseThought,
            ConceptArgument,
            ConceptLexeme,
            SemanticDependency,
        )
        from debilgi.language.proposition_memory import meaning_key
        from debilgi.language.semantic_graph import parse_proposition

        rows = await self._database.fetchall(
            "SELECT gap_id,details_json FROM continuous_cognition_gaps WHERE guild_id=? "
            "AND signature LIKE 'sleep:%' AND status='open' AND source_event_id!=? "
            "ORDER BY last_seen_at DESC LIMIT 32", (event.guild_id, event.event_id))
        plans, seen = [], set()
        for row in rows:
            details = json.loads(row["details_json"])
            if (details.get("actor_id") != event.actor_id
                    or details.get("channel_id") != event.channel_id
                    or '"meaning"' not in (details.get("meaning_scope") or "")[:12]):
                continue
            question = parse_proposition(details.get("prompt") or "", perspective="user",
                                         allow_question=True, possessive_deixis=True)
            subject = next((d.argument.head.lemma for d in question.dependencies
                            if d.role == "subject"), None) if question else None
            if (question is None or subject != "я"
                    or question.predicate.lemma in OWN_STATE_PROJECTIONS):
                continue
            question = replace(question, clause_kind="question")
            key = meaning_key(question)
            if key in seen:
                continue
            seen.add(key)
            thinking = ClauseThought(
                ConceptLexeme(goal.predicate, goal.predicate, "verb", (
                    ("tense", "present"), ("person", "first"), ("number", "singular"))),
                (SemanticDependency("subject", ConceptArgument(ConceptLexeme(
                    "я", "я", "pronoun", (("number", "singular"), ("person", "first"))))),),
                confidence=0.7, evidence=(row["gap_id"],))
            query = ClaimQuery(event.guild_id, event.channel_id, event.actor_id,
                               event.utc_timestamp.isoformat(), event.event_id,
                               predicate=question.predicate.lemma)
            word = next((d.argument.head.lemma for d in question.dependencies
                         if d.argument.head.lemma in {"что", "кто", "как", "где", "куда",
                                                      "почему", "зачем", "когда"}), None)
            plan = MessagePlan(goal, thinking, knowledge_limit=KnowledgeLimit(
                query, question, word), mixture=(("hypothesis", 1.0),))
            plans.append(DerivedPlan(plan, "supposition", (
                SemanticEvidence(row["gap_id"], "open_question", details.get("prompt") or "",
                                 question),), goal.question_word))
            if len(plans) == 2:
                break
        return tuple(plans)

    async def variable_plan(self, event, context):
        """Bind ellipsis against the receipt-scoped proposition or ask its topic."""
        from debilgi.language.canonical_claims import ClaimQuery
        from debilgi.language.communication import KnowledgeLimit, MessagePlan, reader_goal
        from debilgi.language.dialogue_obligations import thought_from_data
        from debilgi.language.generative_planner import DerivedPlan, SemanticEvidence
        from debilgi.language.productive_nlg import (
            ClauseThought,
            ConceptArgument,
            ConceptLexeme,
            SemanticDependency,
        )

        variable = ellipsis_variable(context.prompt)
        if variable is None:
            return None
        prior = context.prior_bot_turn
        limited = bool(variable["role"] == "cause" and prior
                       and prior["plan"].get("knowledge_limit"))
        if variable["role"] == "cause" and prior and prior["plan"].get(
            "claim", {}).get("predicate_class") == "dialogue_act":
            return None  # explain the observed act, not an invented event cause
        if variable["role"] == "cause" and prior and not limited and prior.get("basis"):
            explained = await self.explain_delivered_basis(event, context, prior)
            if explained is not None:
                return explained
        question = None
        receipt = prior["receipt_id"] if prior else None
        if prior and variable["role"] in {"attribute", "cause"}:
            data = prior["plan"]
            # The cause asked after an admitted limit is the cause of the limit itself
            # (the epistemic statement), never of the queried event it did not assert.
            content = (data.get("claim") if limited else
                       (data.get("knowledge_limit") or {}).get("question") or data.get("claim"))
            if (variable["role"] == "attribute" and content
                    and content.get("predicate_class") == "dialogue_act"):
                from debilgi.language.semantic_graph import parse_proposition

                quoted = parse_proposition(prior["response_text"], perspective="agent",
                                           allow_question=True)
                # Re-reading a delivered question supplies variables, never
                # an assertion that its presupposed event actually occurred.
                interrogative = any(c.clause_kind == "question"
                                    for c in exchange_analysis(prior["response_text"])[0].clauses)
                content = (identity_graph(replace(quoted, clause_kind="question"))
                           if quoted and interrogative else None)
            if content and content.get("predicate_class") != "dialogue_act":
                thought = thought_from_data(content)
                if variable["role"] == "attribute":
                    features = variable["features"]
                    def compatible(argument):
                        head = argument.head
                        reading = DEFAULT_MORPHOLOGY.best(head.surface or head.lemma)
                        known = {**(reading.features() if reading else {}), **dict(head.features)}
                        return all(known.get(k) == v for k, v in features.items() if k != "case")
                    candidates = (thought, *(thought_from_data(c)
                                             for c in data.get("additional_claims", ())))
                    matching = [(candidate, d) for candidate in candidates
                                for d in candidate.dependencies if d.role != "subject"
                                and d.argument.head.part_of_speech == "noun"
                                and compatible(d.argument)]
                    if len(matching) == 1:
                        thought, selected = matching[0]
                        argument = replace(selected.argument, modifiers=(ConceptLexeme(
                            "какой", "какой", "adjective", tuple(features.items())),),
                            grammatical_case=features.get("case"))
                        question = replace(thought, clause_kind="question", dependencies=tuple(
                            replace(d, argument=argument) if d == selected else d
                            for d in thought.dependencies), focus_role=selected.role)
                elif thought.clause_kind != "question":
                    question = replace(thought, clause_kind="question", dependencies=(
                        *thought.dependencies, SemanticDependency("cause", ConceptArgument(
                            ConceptLexeme("почему", "почему", "adverb")))), focus_role="cause")
        # Select lexical operators only from original human observations.
        rows = await self._database.fetchall(
            "SELECT d.source_id,d.content FROM language_discourse_turns d "
            "JOIN events e ON e.event_id=d.source_id WHERE d.guild_id=? AND d.observed_at<? "
            "AND COALESCE(json_extract(e.payload_json,'$.author_is_bot'),0)=0 "
            "AND (d.content LIKE '%зна%' OR d.content LIKE '%спраш%') "
            "ORDER BY d.observed_at DESC LIMIT 512",
            (event.guild_id, event.utc_timestamp.isoformat()),
        )
        operations = {}
        for row in rows:
            for token in exchange_analysis(row["content"])[0].tokens:
                if token.lemma in {"знать", "спрашивать"} and token.part_of_speech == "verb":
                    operations.setdefault(token.lemma, row["source_id"])
        goal = reader_goal(context.structure)
        if question and "знать" in operations:
            evidence = (receipt, operations["знать"])
            assertion = ClauseThought(ConceptLexeme("знать", "знать", "verb"), (
                SemanticDependency("subject", ConceptArgument(ConceptLexeme("я", "я", "pronoun"))),
            ), negated=True, evidence=evidence)
            query = ClaimQuery(event.guild_id, event.channel_id, event.actor_id,
                               event.utc_timestamp.isoformat(), event.event_id,
                               predicate=question.predicate.lemma,
                               requested_role=question.focus_role,
                               requested_attribute=variable["role"] == "attribute")
            plan = MessagePlan(goal, assertion, knowledge_limit=KnowledgeLimit(
                query, question, "какой" if variable["role"] == "attribute" else "почему"),
                mixture=(("known", 1.0),), known_slots=((variable["role"], receipt, receipt),))
            return DerivedPlan(plan, "bind_delivered_variable", (
                SemanticEvidence(receipt, "sent_turn", prior["response_text"], question),),
                "почему" if variable["role"] == "cause" else "какой")
        if "спрашивать" not in operations:
            return None
        source = operations["спрашивать"]
        thought = ClauseThought(ConceptLexeme("спрашивать", "спрашивать", "verb", (
            ("tense", "present"),)), (
                SemanticDependency("subject", ConceptArgument(
                    ConceptLexeme("ты", "ты", "pronoun"))),
                SemanticDependency("topic", ConceptArgument(ConceptLexeme("что", "что", "pronoun"),
                                                           preposition="о",
                                                           grammatical_case="prepositional")),
            ), clause_kind="question", focus_role="topic", confidence=0.78 ** 2,
            evidence=(source,))
        return DerivedPlan(MessagePlan(goal, thought, speech_act="clarify_variable",
                                      mixture=(("reaction", 1.0),)), "ask_unbound_variable", (
            SemanticEvidence(source, "human_question_grammar", "", thought),))

    async def sleep_answer_plans(self, event, context, *, breadth=3):
        """Borrow human answer operations, never their personal truth values.

        Exact meaning matches precede slot analogies. The latter can transfer
        an answer's polarity to current slots only as an explicitly marked
        opinion. Corpus authors vote for an operation, not for an agent fact.
        """
        from debilgi.language.lexicon import LexiconService

        rows = await self._database.fetchall(
            "SELECT x.*,p.content prompt,a.content answer,a.actor_id actor_id "
            "FROM language_observed_exchanges x "
            "JOIN language_discourse_turns p ON p.source_id=x.prompt_event_id "
            "JOIN language_discourse_turns a ON a.source_id=x.answer_event_id "
            "JOIN events pe ON pe.event_id=p.source_id JOIN events ae ON ae.event_id=a.source_id "
            "WHERE x.guild_id=? AND x.observed_at<? "
            "AND COALESCE(json_extract(pe.payload_json,'$.author_is_bot'),0)=0 "
            "AND COALESCE(json_extract(ae.payload_json,'$.author_is_bot'),0)=0 "
            "ORDER BY x.observed_at DESC,x.answer_event_id LIMIT 8192",
            (event.guild_id, event.utc_timestamp.isoformat()))
        canonical = await LexiconService.for_meaning(self._database).canonical_lemmas(
            event.guild_id, for_hypothesis=True)
        return await asyncio.to_thread(self._sleep_answer_plans, context,
                                       tuple(dict(r) for r in rows), canonical, breadth)

    @staticmethod
    def _sleep_answer_plans(context, rows, canonical, breadth=3):
        from debilgi.language.communication import MessagePlan, ReaderGoal, reader_goal
        from debilgi.language.dialogue_obligations import thought_from_data
        from debilgi.language.generative_planner import DerivedPlan, SemanticEvidence
        from debilgi.language.productive_nlg import (
            ConceptArgument,
            ConceptLexeme,
            SemanticDependency,
        )
        from debilgi.language.proposition_memory import count_meaning_evidence, meaning_key
        from debilgi.language.request_focus import polarity_of
        from debilgi.language.semantic_graph import parse_proposition

        goal = reader_goal(context.structure)
        target = parse_proposition(context.structure, allow_question=True)
        if target is None:
            # «будешь молчать» asks about the infinitive's event. The reading is
            # opt-in, so daytime readers keep the auxiliary as the predicate.
            target = parse_proposition(context.structure, allow_question=True,
                                       future_auxiliary=True)
            if target is not None and goal.predicate != target.predicate.lemma:
                goal = replace(goal, predicate=target.predicate.lemma)
            if target is not None and any(
                    t.part_of_speech == "verb" and dict(t.grammatical_features).get(
                        "person") == "first" and dict(t.grammatical_features).get(
                        "number") == "plural" for t in context.structure.tokens):
                # An inclusive «будем» keeps the bot among the participants.
                target = replace(target, dependencies=tuple(
                    replace(d, argument=replace(d.argument, head=replace(
                        d.argument.head, lemma="мы", surface="мы")))
                    if d.role == "subject" and d.argument.head.lemma == "вы" else d
                    for d in target.dependencies))
        opinion_request = goal.predicate in {"предположить", "прикинуть"}
        prior = context.prior_bot_turn
        if opinion_request and prior:
            limit = prior["plan"].get("knowledge_limit") or {}
            if limit.get("question"):
                target = thought_from_data(limit["question"])
                goal = ReaderGoal(**prior["plan"]["goal"])
        if target is None:
            return ()
        target_key = meaning_key(target, canonical)
        # The exchange may contain a greeting or a second question. Read the
        # actual clause boundaries instead of requiring a one-sentence message.
        expanded = []
        for row in rows:
            structure = exchange_analysis(row["prompt"])[0]
            for clause in structure.clauses:
                tokens = structure.tokens[clause.token_start:clause.token_end]
                if clause.clause_kind == "question":
                    expanded.append({**row, "prompt": " ".join(t.surface for t in tokens)})
        answer_rows = rows
        rows = expanded
        operators, grouped = {}, {}
        for row in answer_rows:
            for token in exchange_analysis(row["answer"])[0].tokens:
                if token.lemma in {"наверное", "возможно", "вероятно"}:
                    operators.setdefault(token.lemma, []).append(row)
        if not operators:
            return ()

        def support(items):
            return count_meaning_evidence({
                "actor_id": r["actor_id"], "observed_at": r["observed_at"], "polarity": 1,
                "source_event_id": r["answer_event_id"],
                # These votes license language use, not canonical knowledge.
                "source": "human_answer_operation",
            } for r in items)

        ranked_operators = sorted(operators.items(), key=lambda item: (
            support(item[1]).people, item[0]), reverse=True)
        operator, sources = ranked_operators[0]
        operator_source = sources[0]["answer_event_id"]
        # Hedges several different people used are all learned forms of the same
        # operation; the first (strongest) stays the default.
        hedges = [(name, rows_[0]["answer_event_id"]) for name, rows_ in ranked_operators
                  if support(rows_).people >= 2 or name == operator]
        for row in rows:
            old_goal = reader_goal(exchange_analysis(row["prompt"])[0])
            if (canonical.get(old_goal.predicate, old_goal.predicate) !=
                    canonical.get(goal.predicate, goal.predicate)
                    or old_goal.question_word != goal.question_word
                    or old_goal.addressed != goal.addressed):
                continue
            old = parse_proposition(row["prompt"], allow_question=True)
            if old is None:
                continue
            exact = meaning_key(old, canonical) == target_key
            # Generalize argument values, but retain the requested predicate,
            # participant direction, tense and the set of semantic roles.
            if not exact and ({d.role for d in old.dependencies} !=
                              {d.role for d in target.dependencies}
                              or dict(old.predicate.features).get("tense", "present") !=
                              dict(target.predicate.features).get("tense", "present")):
                continue
            answer_structure = exchange_analysis(row["answer"])[0]
            first = answer_structure.clauses[0] if answer_structure.clauses else None
            answer_text = (" ".join(t.surface for t in answer_structure.tokens[
                first.token_start:first.token_end]) if first else row["answer"])
            answer = parse_proposition(answer_text, perspective="agent")
            words = [t for t in exchange_analysis(answer_text)[0].tokens
                     if t.surface[:1].isalnum()]
            polar = next((polarity_of(t.lemma) for t in words
                          if polarity_of(t.lemma) is not None), None)
            if (answer is None and goal.question_word == "что"
                    and any(t.lemma in {"ничто", "ничего"} for t in words)
                    and all(t.part_of_speech in _CLOSED | {"interjection"} for t in words)):
                answer = replace(target, negated=True, dependencies=(
                    *target.dependencies, SemanticDependency("direct_object", ConceptArgument(
                        ConceptLexeme("ничто", "ничего", "pronoun"),
                        grammatical_case="genitive"))))
            if answer is None and not goal.question_word:
                finite = [t for t in words if t.part_of_speech == "verb"]
                if len(finite) == 1 and finite[0].lemma == old.predicate.lemma:
                    # A finite "want / don't want" is an elliptical polar
                    # answer, not an unbound action to copy into a new episode.
                    polar = not any(t.lemma == "не" for t in words)
            if answer is None and polar is None:
                continue
            if goal.question_word:
                if answer is None:
                    continue
                if goal.addressed:
                    # What the agent itself does is not another person's answer: copying a
                    # donor's whole proposition would hand it their personal experience.
                    # Their answers lend an operation (the hedge), never this content.
                    continue
                # An answer to an open action slot may choose a specific
                # predicate. It must still speak about the requested person.
                subject = next((d.argument.head.lemma for d in answer.dependencies
                                if d.role == "subject"), None)
                expected = "я" if goal.addressed else "ты"
                if subject != expected:
                    continue
                if goal.predicate != "делать" and answer.predicate.lemma != goal.predicate:
                    continue
                thought = answer
            else:
                if answer is not None:
                    if answer.predicate.lemma != old.predicate.lemma:
                        continue
                    old_subject = next((d.argument.head.lemma for d in answer.dependencies
                                        if d.role == "subject"), None)
                    if old_subject != ("я" if goal.addressed else "ты"):
                        continue
                    polar = not answer.negated
                if polar is None:
                    continue
                # The donor chooses an operation, the current query owns slots.
                thought = replace(target, negated=not polar)
            thought = replace(thought, clause_kind="statement", focus_role=None,
                              confidence=.45, evidence=(row["answer_event_id"], operator_source))
            key = (meaning_key(thought, canonical), thought.negated)
            group = grouped.setdefault(key, {"thought": thought, "rows": [], "match": 0.0})
            group["rows"].append(row)
            group["match"] = max(group["match"], 1.0 if exact else .75)
        composed = False
        if not grouped and not goal.question_word:
            # No human answered this meaning. Cross the question's own thought
            # (a predicate and its participants) with the learned epistemic
            # operator and each polarity: unconfirmed, marked guesses.
            composed = True
            for negated in (False, True):
                thought = replace(target, negated=negated, clause_kind="statement",
                                  focus_role=None, confidence=.4,
                                  evidence=(context.source_event_id, operator_source))
                grouped[(meaning_key(thought, canonical), negated)] = {
                    "thought": thought, "rows": [], "match": 0.5}
        target_terms = {d.argument.head.lemma for d in target.dependencies}
        ranked = sorted(grouped.values(), key=lambda g: (
            g["match"], support(g["rows"]).people,
            int(canonical.get(g["thought"].predicate.lemma, g["thought"].predicate.lemma)
                == canonical.get(goal.predicate, goal.predicate)),
            -sum(d.argument.head.part_of_speech in {"noun", "adjective"}
                 and d.argument.head.lemma not in target_terms
                 for d in g["thought"].dependencies),
            meaning_key(g["thought"], canonical)),
            reverse=True)
        result = []
        # Every meaning first in the strongest hedge, then the same meanings in the others.
        options = [(group, name, source) for name, source in hedges for group in ranked]
        for group, operator, operator_source in options[:breadth]:
            donors = group["rows"]
            thought = replace(group["thought"], evidence=tuple(dict.fromkeys(
                [*(group["thought"].evidence[:1] if composed else ()),
                 *(r["answer_event_id"] for r in donors), operator_source])))
            plan = MessagePlan(goal, thought, speech_act="opinion", hypothesis=(
                ("kind", "experience_opinion"), ("operator", operator),
                ("operator_source", operator_source)), mixture=(("hypothesis", 1.0),))
            proposal = DerivedPlan(plan, "own_thought_crossover" if composed
                                   else "human_answer_crossover", tuple(
                SemanticEvidence(r["answer_event_id"], "human_answer_operation",
                                 r["answer"], thought)
                for r in donors[:8]))
            result.append((proposal, {
                "sleep_opinion": True, "sleep_opinion_request": opinion_request,
                "sleep_query_goal": asdict(goal), "sleep_canonical": canonical,
                "sleep_query_key": target_key,
                "sleep_answer_slot": bool(goal.question_word),
                "human_answer_people": support(donors).people,
                "human_answer_match": group["match"],
                "human_answer_sources": list(thought.evidence),
                "sleep_composed": composed,
            }))
        return tuple(result)

    async def sleep_repair_plan(self, event, context, neighbours):
        """Resolve a reasked element of a delivered turn against that episode."""
        from debilgi.language.canonical_claims import ClaimQuery
        from debilgi.language.communication import KnowledgeLimit, MessagePlan, reader_goal
        from debilgi.language.dialogue_state import analyze_dialogue_state
        from debilgi.language.generative_planner import DerivedPlan, SemanticEvidence
        from debilgi.language.productive_nlg import (
            ClauseThought,
            ConceptArgument,
            ConceptLexeme,
            SemanticDependency,
            unresolved_answer_thought,
        )

        prior = context.prior_bot_turn
        if not prior:
            return None
        goal = reader_goal(context.structure)
        old_goal = reader_goal(exchange_analysis(prior["response_text"])[0])
        if not (event.turn_form.kind == "repair" or goal.question_word and
                goal.predicate == old_goal.predicate and not goal.subject):
            return None
        episode = tuple(str(n.payload.get("content", "")) for n in neighbours
                        if n.event_id != event.event_id and not n.payload.get("author_is_bot")
                        and n.actor_id == event.actor_id and n.utc_timestamp < event.utc_timestamp)
        state = analyze_dialogue_state(prior["response_text"], episode)
        references = [r for r in state.references if r.kind == "explicit_pronoun"]
        if not references:
            return None
        reference = references[0]
        receipt = prior["receipt_id"]
        subject = SemanticDependency("subject", ConceptArgument(
            ConceptLexeme("я", "я", "pronoun")))
        if reference.chosen:
            referent = reference.chosen
            thought = ClauseThought(ConceptLexeme("говорить", "говорить", "verb", (
                ("tense", "past"), ("gender", "masculine"))), (
                    subject, SemanticDependency("topic", ConceptArgument(
                        ConceptLexeme(referent.lemma, referent.surface, "noun"),
                        preposition="о", grammatical_case="prepositional"))),
                confidence=.7, evidence=(receipt,))
            plan = MessagePlan(goal, thought)
        else:
            # The receipt proves that we said it, not that its missing referent
            # existed. Express the limit about the intended utterance itself.
            question = ClauseThought(ConceptLexeme("хотеть", "хотеть", "verb", (
                ("tense", "past"), ("gender", "masculine"))), (
                    subject, SemanticDependency("action_complement", ConceptArgument(
                        ConceptLexeme("сказать", "сказать", "verb"))),
                    SemanticDependency("direct_object", ConceptArgument(
                        ConceptLexeme("что", "что", "pronoun")))),
                clause_kind="question", predicate_class="modal", focus_role="direct_object",
                confidence=.7, evidence=(receipt,))
            assertion = unresolved_answer_thought(confidence=.7)
            assertion = replace(assertion, dependencies=(subject,), evidence=(receipt,))
            query = ClaimQuery(event.guild_id, event.channel_id, event.actor_id,
                               event.utc_timestamp.isoformat(), event.event_id,
                               predicate="сказать", requested_role="direct_object")
            plan = MessagePlan(goal, assertion, knowledge_limit=KnowledgeLimit(
                query, question, "что"))
        return (DerivedPlan(plan, "explain_delivered_reference", (
            SemanticEvidence(receipt, "sent_turn", prior["response_text"], plan.claim),)), {
                "repair_unbound_reference": reference.chosen is None,
                "repair_reference_slot": reference.surface,
                "repair_reference_binding": reference.chosen.lemma if reference.chosen else None,
                "sleep_answer_slot": True,
            })

    async def memory_activation(self, event: Event, structure):
        """Reuse canonical argument/claim and observed predicate-parent edges.

        These currents are linguistic neighbourhoods, never facts about the
        current speaker. Only original evidence edges can propagate, depth <= 3.
        """
        from debilgi.language.canonical_claims import ClaimQuery
        from debilgi.language.semantic_graph import spread_activation

        terms = sorted({t.lemma for t in structure.tokens
                        if t.part_of_speech in {"noun", "verb"}})[:12]
        if not terms:
            return {}, ()
        placeholders = ",".join("?" for _ in terms)
        rows = await self._database.fetchall(
            "SELECT DISTINCT c.claim_id, c.predicate, c.source_event_id, a.ref "
            "FROM meaning_claims c JOIN meaning_arguments a ON a.claim_id = c.claim_id "
            "WHERE c.guild_id = ? AND c.source_time < ? AND c.resolved = 1 "
            "AND c.polarity = 1 AND c.modality_json = '[]' AND "
            f"(c.predicate IN ({placeholders}) OR a.ref IN ({placeholders})) "
            "ORDER BY c.source_time DESC LIMIT 64",
            (event.guild_id, event.utc_timestamp.isoformat(), *terms,
             *("concept:" + term for term in terms)),
        )
        edges, sources = [], []
        for row in rows:
            predicate, argument = "predicate:" + row["predicate"], row["ref"]
            edges.extend(((predicate, row["claim_id"], 0.5),
                          (argument, row["claim_id"], 0.5),
                          (row["claim_id"], predicate, 0.5)))
            sources.append(row["source_event_id"])
        children, rejected = await self.interlocutor.predicate_children(ClaimQuery(
            event.guild_id, event.channel_id, event.actor_id, event.utc_timestamp.isoformat(),
            event.event_id, predicate_parent="делать",
        ))
        for predicate, ids in children.items():
            if predicate not in rejected:
                edges.extend((("predicate:" + predicate, "predicate_parent:делать", 0.5),
                              ("predicate_parent:делать", "predicate:" + predicate, 0.3)))
                if predicate in terms:
                    sources.extend(ids)
        seeds = {prefix + term: 1.0 for term in terms for prefix in ("predicate:", "concept:")}
        return spread_activation(seeds, edges), tuple(dict.fromkeys(sources))

    async def hypothesis_allowed(self, event: Event) -> bool:
        """Only actual deliveries spend the one-hypothesis interlocutor budget, within the episode
        window (as for recent reactions): an old guess does not forbid a free reaction forever."""
        row = await self._database.fetchone(
            "SELECT prompt_event_id,selection_json FROM language_meaning_intentions "
            "WHERE guild_id=? AND channel_id=? AND user_id=? AND sent_at < ? "
            "AND sent_at >= ? AND response_message_id IS NOT NULL ORDER BY sent_at DESC LIMIT 1",
            (event.guild_id, event.channel_id, event.actor_id, event.utc_timestamp.isoformat(),
             (event.utc_timestamp - timedelta(minutes=10)).isoformat()))
        if row is None:
            return True
        decision = json.loads(row["selection_json"]).get("response_decision", {})
        if not decision:
            prior = await self._database.fetchone(
                "SELECT s.details_json FROM language_response_traces t "
                "JOIN language_response_trace_steps s ON s.trace_id=t.trace_id "
                "WHERE t.source_event_id=? AND s.stage='intent' AND s.component='response_mode' "
                "ORDER BY s.sequence DESC LIMIT 1", (row["prompt_event_id"],))
            decision = json.loads(prior[0]).get("decision", {}) if prior else {}
        return decision.get("confidence_level") != "hypothesis"

    async def observe_hypothesis_feedback(self, event: Event) -> bool:
        from debilgi.experience.feedback import DEFAULT_FEEDBACK_ATTRIBUTOR

        feedback = DEFAULT_FEEDBACK_ATTRIBUTOR.attribute(event)
        if feedback.target_message_id is None:
            return False
        prior = await self._database.fetchone(
            """SELECT s.details_json, p.content, t.trace_id, i.prompt_event_id
               FROM language_meaning_intentions i
               JOIN language_response_traces t ON t.source_event_id = i.prompt_event_id
               JOIN language_response_trace_steps s ON s.trace_id = t.trace_id
                 AND s.component = 'response_mode' AND s.stage = 'intent'
               JOIN language_discourse_turns p ON p.source_id = i.prompt_event_id
               WHERE i.guild_id = ? AND i.channel_id = ? AND i.user_id = ?
                 AND i.response_message_id = ? AND i.sent_at < ? AND i.sent_at >= ?
               LIMIT 1""",
            (event.guild_id, event.channel_id, event.actor_id, feedback.target_message_id,
             event.utc_timestamp.isoformat(),
             (event.utc_timestamp - timedelta(minutes=10)).isoformat()),
        )
        if prior is None:
            return False
        decision = json.loads(prior["details_json"]).get("decision", {})
        if decision.get("confidence_level") != "hypothesis" or not decision.get("rule_id"):
            return False
        selected = await self._database.fetchone(
            "SELECT details_json FROM language_response_trace_steps WHERE trace_id = ? "
            "AND stage = 'selection' AND component = 'response_arbitrator' "
            "ORDER BY sequence DESC LIMIT 1", (prior["trace_id"],),
        )
        if selected:
            actual = (json.loads(selected[0]).get("diagnostic") or {}).get("reaction_rule")
            if actual and actual.get("rule_id"):
                decision = {**decision, **actual}
        reward = feedback.reward if feedback.is_learnable() else 0.0
        if not reward and feedback.kind == "conversation_continued":
            # Topic uptake is weaker evidence than an explicit confirmation.
            current = {f for f in turn_features(str(event.payload.get("content", "")))
                       if f.startswith(("predicate:", "topic:"))}
            if current & turn_features(prior["content"]):
                reward = 0.25
        if not reward:
            return False
        rows = await self._database.fetchall(
            "SELECT answer_event_id, topic_links_json FROM language_observed_exchanges "
            "WHERE guild_id = ?", (event.guild_id,),
        )
        updates = []
        target = prior["prompt_event_id"]
        if any(target in link.get("feedback_targets", ()) for row in rows
               for link in json.loads(row["topic_links_json"])
               if link.get("rule_id") == decision["rule_id"]):
            return False  # one attributed reaction per delivered hypothesis
        for row in rows:
            links = json.loads(row["topic_links_json"])
            changed = False
            for link in links:
                if link.get("rule_id") != decision["rule_id"] or event.event_id in link.get(
                    "feedback_ids", ()
                ):
                    continue
                link["weight"] = round(max(0.05, min(1.0, link["weight"] + reward * 0.15)), 6)
                link["support"] += int(reward > 0)
                link["feedback_ids"].append(event.event_id)
                link.setdefault("feedback_targets", []).append(target)
                if reward < 0:
                    signature = "|".join(sorted(decision.get("input_features", ())))
                    if signature not in link["exceptions"]:
                        link["exceptions"].append(signature)
                changed = True
            if changed:
                updates.append((json.dumps(links, ensure_ascii=False, sort_keys=True),
                                row["answer_event_id"]))
        if updates:
            self._exchange_revision += 1
            async with self._database.transaction() as connection:
                await connection.executemany(
                    "UPDATE language_observed_exchanges SET topic_links_json = ? "
                    "WHERE answer_event_id = ?", updates,
                )
        return bool(updates)

    async def turn_form_for(self, event: Event):
        """Use only this interlocutor's earlier manners and the bot reply edge."""
        rows = await self._database.fetchall(
            """SELECT content, actor_id, observed_at FROM language_discourse_turns
               WHERE guild_id = ? AND channel_id = ? AND observed_at < ?
                 AND (actor_id = ? OR (actor_id = ? AND addressed_to_id = ?))
               ORDER BY observed_at DESC LIMIT 32""",
            (event.guild_id, event.channel_id, event.utc_timestamp.isoformat(),
             event.actor_id, event.payload.get("bot_user_id"), event.actor_id),
        )
        human = [analyze_turn_form(r["content"]) for r in rows if r["actor_id"] == event.actor_id]
        habitual = tuple(sorted(t for t in {t for f in human for t in f.tone}
                                if sum(t in f.tone for f in human) / len(human) >= 0.5))
        prior = next((r["content"] for r in rows
                      if r["actor_id"] == event.payload.get("bot_user_id")
                      and r["observed_at"] >= (
                          event.utc_timestamp - timedelta(minutes=10)).isoformat()), "")
        if not prior:
            receipt = await self.prior_response_for(event)
            prior = receipt["response_text"] if receipt else ""
        form = event.turn_form
        return analyze_turn_form(str(event.payload.get("content", "")),
                                 addressed=form.addressed, address_evidence=form.address_evidence,
                                 follows_bot=form.follows_bot or bool(prior),
                                 previous_bot_text=prior,
                                 habitual_tone=habitual)

    async def observe(self, event: Event) -> None:
        if (
            event.event_type not in _MESSAGE_TYPES
            or event.channel_id is None
            or event.actor_id is None
            or event.payload.get("author_is_bot")
        ):
            return
        current = await self._database.fetchone(
            "SELECT * FROM language_discourse_turns WHERE source_id = ?", (event.event_id,)
        )
        if current is None:
            return
        await self.observe_hypothesis_feedback(event)
        # Explicit reply edges win. For ordinary chat, adjacency is bounded by
        # time and participant direction; same-speaker continuations aren't pairs.
        parent_id = current["parent_message_id"]
        parent = await self._database.fetchone(
            """SELECT * FROM language_discourse_turns
               WHERE guild_id = ? AND channel_id = ? AND observed_at < ?
                 AND observed_at >= ? AND source_id != ?
                 AND (? IS NULL OR message_id = ?)
               ORDER BY observed_at DESC, source_id LIMIT 1""",
            (
                event.guild_id,
                event.channel_id,
                current["observed_at"],
                (event.utc_timestamp - timedelta(minutes=5)).isoformat(),
                event.event_id,
                parent_id,
                parent_id,
            ),
        )
        structure = event.meaning
        # This used to run only while generating a reply. Every observed turn
        # can teach productive syntax, including turns that don't address us.
        await self._assimilation.observe_local_structure(
            guild_id=event.guild_id,
            structure=structure,
            observed_at=event.utc_timestamp,
            source_event_id=event.event_id,
        )
        if parent is None or parent["actor_id"] in {None, event.actor_id}:
            return
        # Never treat the bot's output as human demonstration data.
        if parent["actor_id"] == event.payload.get("bot_user_id"):
            return
        parent_event = await self._database.fetchone(
            "SELECT payload_json FROM events WHERE event_id = ?", (parent["source_id"],),
        )
        if parent_event is None or json.loads(parent_event["payload_json"]).get("author_is_bot"):
            return
        await self.interlocutor.observe_continuation(event, parent, current)
        await self.interlocutor.observe_pair(event, parent, current)
        _, prompt_act, features = exchange_analysis(parent["content"])
        prompt_act = reactive_act(parent["content"]) or prompt_act
        _, response_act, _ = exchange_analysis(current["content"])
        await self._persist_analysis(parent["content"], current["content"])
        forms, links = cached_response_forms(parent["content"], current["content"])
        links = [dict(link, source_ids=[parent["source_id"], event.event_id])
                 if link.get("kind") == "feature_transformation" else link for link in links]
        register = reaction_register(parent["content"])
        register_label = "coarse" if "coarse" in register else (
            "formal" if "formal" in register else "informal" if "smile" in register else "neutral"
        )
        async with self._database.transaction() as connection:
            cursor = await connection.execute(
                """INSERT OR IGNORE INTO language_observed_exchanges VALUES
                   (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.event_id,
                    parent["source_id"],
                    event.guild_id,
                    event.channel_id,
                    prompt_act,
                    response_act,
                    json.dumps(sorted(features), ensure_ascii=False),
                    json.dumps(forms, ensure_ascii=False, sort_keys=True),
                    json.dumps(links, ensure_ascii=False, sort_keys=True),
                    event.utc_timestamp.isoformat(),
                ),
            )
            if cursor.rowcount:
                await _store_part_constructions(
                    connection, event.guild_id,
                    {f["construction_id"]: (f, 1) for f in forms if f.get("construction_id")},
                    event.utc_timestamp.isoformat(), increment=True,
                )
                await connection.execute(
                    """INSERT INTO language_dialogue_act_patterns
                       (guild_id, prompt_act, response_act, register_label, confidence,
                        source, updated_at) VALUES (?, ?, ?, ?, 0.70,
                        'local_exchange', ?)
                       ON CONFLICT (guild_id, prompt_act, response_act, register_label, source)
                       DO UPDATE SET evidence_count = evidence_count + 1,
                         confidence = MIN(0.95, confidence + 0.005),
                         updated_at = excluded.updated_at""",
                    (event.guild_id, prompt_act, response_act, register_label,
                     event.utc_timestamp.isoformat()),
                )

    async def recent_reactions(
        self, event: Event, *, minutes: int = 10,
    ) -> tuple[tuple[str, str | None, bool], ...]:
        """Receipt-backed episode history, scoped to the current interlocutor."""
        rows = await self._database.fetchall(
            """SELECT i.response_text,
                      json_extract(i.selection_json, '$.exchange_pattern_id') AS pattern,
                      instr(i.response_text, '?') > 0 AS question
               FROM language_meaning_intentions i
               WHERE i.guild_id = ? AND i.channel_id = ? AND i.user_id = ?
                 AND i.sent_at >= ? AND i.sent_at < ? AND i.response_message_id IS NOT NULL
               ORDER BY i.sent_at DESC, i.intention_id LIMIT 6""",
            (event.guild_id, event.channel_id, event.actor_id,
             (event.utc_timestamp - timedelta(minutes=minutes)).isoformat(),
             event.utc_timestamp.isoformat()),
        )
        return tuple((r["response_text"], r["pattern"], bool(r["question"]))
                     for r in rows if r["response_text"])

    async def source_answers(self, event: Event) -> frozenset[str]:
        rows = await self._cached_rows("answers", event,
            """SELECT a.content FROM language_discourse_turns a
               JOIN events e INDEXED BY events_human_answers ON e.event_id = a.source_id
               WHERE a.guild_id = ? AND a.observed_at < ?
                 AND COALESCE(json_extract(e.payload_json, '$.author_is_bot'), 0) = 0
                 AND (e.source LIKE 'seed_corpus_%' OR EXISTS (
                   SELECT 1 FROM language_observed_exchanges x
                   WHERE x.answer_event_id = a.source_id))
               ORDER BY a.observed_at DESC, a.source_id LIMIT 16384""",
            (event.guild_id, event.utc_timestamp.isoformat()),
        )
        return frozenset(normalized_response(r["content"]) for r in rows)

    async def forms_for(self, event: Event, prompt: str, *, accepting_report: bool = False,
                        hostile: bool = False) -> tuple[ObservedDialogueForm, ...]:
        await self._load_analysis_cache()
        _, act, features = exchange_analysis(prompt)
        reaction = ("acknowledgment" if accepting_report else
                    reactive_act(prompt) or analyze_turn_form(prompt).kind)
        register = reaction_register(prompt)
        structure = exchange_analysis(prompt)[0]
        memory_currents, activation_sources = await self.memory_activation(event, structure)
        valence = lexical_valence(prompt)
        if hostile and valence is None:
            valence = -1.0  # an attack on the addressee (cognition.affect.hostility_cues)
        condition = reaction == "self_report" and any(
            c.predicate_index is not None
            and dict(structure.tokens[c.predicate_index].grammatical_features).get("transitivity")
                == "intransitive" and not c.direct_object_indices
            and not c.action_complement_indices
            for c in structure.clauses)
        rows = await self._cached_rows("forms", event,
            """SELECT x.*, p.content AS prompt, a.content AS answer, p.actor_id AS prompt_actor_id
               FROM language_observed_exchanges x
               JOIN language_discourse_turns p ON p.source_id = x.prompt_event_id
               JOIN language_discourse_turns a ON a.source_id = x.answer_event_id
               WHERE x.guild_id = ? AND x.observed_at < ?
               ORDER BY x.observed_at DESC, x.answer_event_id LIMIT 8192""",
            (event.guild_id, event.utc_timestamp.isoformat()),
        )
        if valence is None and reaction in {"evaluation", "fragment"}:
            valence, evaluation_sources = learned_evaluation(prompt, rows)
            activation_sources = tuple(dict.fromkeys((*activation_sources, *evaluation_sources)))
        target = evaluation_target(prompt)
        if reaction == "evaluation":
            exchange_value, exchange_target, sources = learned_evaluation_exchange(prompt, rows)
            if valence is None:
                valence = exchange_value
            elif exchange_value is not None and valence * exchange_value < 0:
                valence = None  # incompatible human interpretations remain unresolved
            target = target or exchange_target
            activation_sources = tuple(dict.fromkeys((*activation_sources, *sources)))
        specialized = computed_question_forms(event, prompt, rows)
        if reaction == "acknowledgment":
            specialized += evaluation_tone_forms(prompt, rows, reaction=reaction)

        if reaction == "evaluation" and valence is None:
            specialized += evaluation_tone_forms(prompt, rows)
        if reaction is not None and event.turn_form.addressed:
            specialized += evaluation_tone_forms(prompt, rows, reaction=reaction, recovery=True)
        if reaction == "evaluation":
            specialized += heard_question_forms(event, prompt, rows)
        if valence is None and reaction in {"self_report", "emotion"}:
            predicates = {structure.tokens[c.predicate_index].lemma for c in structure.clauses
                          if c.predicate_index is not None}
            evidence = []
            for row in rows:
                source_structure = exchange_analysis(row["prompt"])[0]
                for clause in source_structure.clauses:
                    if clause.predicate_index is None or source_structure.tokens[
                        clause.predicate_index
                    ].lemma not in predicates:
                        continue
                    if source_structure.tokens[clause.predicate_index].lemma == "быть":
                        continue  # a copula alone carries no event valence
                    # An adjective attached to a different participant/object
                    # does not appraise this predicate. Transfer the predicate's
                    # own polarity, or a direct short human event appraisal.
                    value = lexical_valence(source_structure.tokens[clause.predicate_index].surface)
                    answer = exchange_analysis(row["answer"])[0]
                    if value is None and len(answer.clauses) == 1 and len(answer.tokens) <= 6 \
                            and not any(c.predicate_index is not None for c in answer.clauses) \
                            and not any(t.lemma in {"я", "мы"} for t in answer.tokens):
                        value = lexical_valence(row["answer"])
                        if value is not None and clause.negated:
                            value = -value
                    # An enthusiastic reply alone cannot establish the valence
                    # of an unfamiliar state. A raw negative appraisal vetoes
                    # a numerical majority of enthusiastic pairs.
                    if value is not None and (value < 0 or not condition):
                        evidence.append(value)
            if evidence:
                negative = [v for v in evidence if v < 0]
                valence = min(negative) if negative else sum(evidence) / len(evidence)
                if any(c.negated for c in structure.clauses):
                    valence = -valence
        concern = (reaction == "emotion" or condition) and valence is None \
            or valence is not None and valence < 0
        patterns = await self._database.fetchall(
            "SELECT prompt_act, response_act FROM language_dialogue_act_patterns "
            "WHERE guild_id = ? AND evidence_count > 0", (event.guild_id,),
        )
        witnessed_acts = {(r["prompt_act"], r["response_act"]) for r in patterns}
        current_features = set(turn_features(prompt))
        current_features.add("addressed:" + str(event.turn_form.addressed))
        if target:
            current_features.add("evaluation_target:" + target)
        if valence is not None:
            current_features.add("valence:" + ("negative" if valence < 0 else "positive"))
        content = {t.lemma for t in structure.tokens
                   if t.part_of_speech in {"noun", "verb", "adjective"}
                   and t.semantic_role not in {
                       "pragmatic_marker", "question_target", "time", "duration"}}
        upgrades, constructions, rule_support = rows.projection_totals()
        grouped, rule_data = {}, {}
        for row_index, row in enumerate(rows.form_rows(witnessed_acts)):
            if row_index % 128 == 0:
                await asyncio.sleep(0)
            stored, rules = row["_stored"], row["_rules"]
            _, old_act, previous = row["_analysis"]
            old_reaction = row["_reaction"]
            common = features & previous
            old_content = row["_content"]
            content_overlap = bool((content & old_content) - {"быть"})
            predicate_overlap = bool(
                {f for f in features & previous if f.startswith("lemma:")}
                & {"lemma:" + t.lemma for t in structure.tokens
                   if t.part_of_speech == "verb" and t.lemma != "быть"}
            )
            relevance = 0.0
            same_act = act == old_act or reaction and reaction == old_reaction
            for form in stored:
                if not form.get("meaning"):
                    continue  # legacy factual slots aren't a reaction source
                component, meaning = form["component"], form["meaning"]
                directed_retort = meaning == "retort" and (
                    valence is not None and valence < 0 and target == "addressee"
                )
                # A directly witnessed return on the same content remains a
                # reaction, even when its sign is unknown. It cannot label a
                # new word as hostile or license a cross-content attack.
                witnessed_return = (meaning == "retort" and valence is None
                                    and content_overlap and same_act
                                    and (target == "addressee" or not any(
                                        c.subject_indices or c.experiencer_indices
                                        for c in structure.clauses))
                                    and (evaluation_target(row["prompt"]) == "addressee"
                                         or not any(c.subject_indices or c.experiencer_indices
                                                    for c in exchange_analysis(
                                                        row["prompt"])[0].clauses)))
                if meaning == "retort" and not (directed_retort or witnessed_return):
                    continue
                # The transformation has semantic argument types. A laugh
                # witnessed after one question is not a rule for all questions;
                # turn shape alone cannot stand in for the joke/event content.
                if component == "core":
                    if reaction in {"question", "request"} and meaning in {
                        "appraise", "acknowledge", "retort", "mirror_amusement", "return_greeting",
                    }:
                        continue  # the requested proposition has not been supplied
                    if meaning == "mirror_amusement" and reaction != "amusement" \
                            and not content_overlap:
                        continue
                    if (form.get("response_turn") in {"amusement", "emotion"}
                            and reaction != form["response_turn"] and not content_overlap):
                        continue
                    if (meaning == "acknowledge" and reaction in {"question", "request", "fragment"}
                            and not content_overlap):
                        continue
                    if (meaning == "reciprocal_question" and reaction == "question"
                            and not content_overlap):
                        continue
                    if (meaning == "reciprocal_question" and reaction in {"fragment", "request"}
                            and not content_overlap):
                        continue
                if (reaction == "smalltalk" and meaning == "return_greeting"
                        and not analyze_pragmatics(prompt).functions & {"greeting"}
                        and not event.turn_form.follows_bot):
                    continue
                rule = rules.get(form.get("rule_id"))
                if component == "core":
                    if rule is None:
                        continue
                    activation = activate_transformation(
                        rule, current_features,
                        memory=memory_currents,
                        bridge=(meaning in {"reciprocal_question", "empathize"} and concern
                                or meaning == "slot_question"
                                and reaction in {"self_report", "fragment"}
                                or directed_retort and reaction == "evaluation"
                                and "turn:negative_evaluation" in rule["input"]),
                    )
                    if meaning == "express_limit":
                        continue  # factual Unknown belongs to the scoped claim reader
                    if specialized and reaction in {"self_report", "emotion"} and meaning in {
                        "reciprocal_question", "check_condition", "appraise", "empathize",
                    }:
                        continue
                    if (meaning == "acknowledge" and reaction == "self_report"
                            and lexical_valence(prompt) is None):
                        activation = min(0.25, activation)
                    # An ungrounded emotion fragment cannot become an asserted
                    # state (e.g. somebody else's 'nothing much').
                    appraisal = lexical_valence(" ".join(s["surface"] for s in form["slots"]))
                    if meaning == "appraise":
                        tentative = (valence is None and appraisal is None
                                     and predicate_overlap
                                     and all(s["binding"] == "speech_act" for s in form["slots"]))
                        if valence is None and not tentative or (
                            valence is not None and appraisal is not None
                            and appraisal * valence <= 0
                        ):
                            continue
                        if tentative:
                            activation = min(0.35, activation)
                        if lexical_valence(prompt) is None:
                            # An appraisal inferred from somebody else's reply
                            # is weaker than a transformation binding this event's
                            # actual argument. Frequency cannot supply its valence.
                            activation = min(0.28, activation)
                    if activation <= 0:
                        continue
                    relevance = activation
                    rule_key = json.dumps(form, sort_keys=True)
                    if activation > rule_data.get(rule_key, ({}, -1.0))[1]:
                        rule_data[rule_key] = rule, activation
                else:
                    relevance = 0.5
                if reaction == "fragment" and meaning == "slot_question" and not any(
                    s.get("source_role") == "unfamiliar_word" for s in form["slots"]
                ) and len([t for t in structure.tokens if t.part_of_speech == "noun"
                           and t.semantic_role not in {"time", "duration", "vocative"}]) != 1:
                    continue
                if reaction == "fragment" and meaning == "slot_question" and not any(
                    s["binding"] == "prompt" for s in form["slots"]
                ) and not event.turn_form.follows_bot:
                    continue  # a detached adjective needs an actual antecedent
                if reaction == "smalltalk" and meaning == "reciprocal_question" \
                        and form.get("question_slot") not in {None, "как"}:
                    continue
                if meaning == "retort" and reaction == "fragment" and not any(
                    f.startswith("lemma:") for f in common
                ) and "coarse" not in register:
                    # An unmarked unfamiliar fragment is not evidence of an
                    # attack. Insulting retorts need an evaluative/address cue.
                    continue
                # Tone is transferable language. Cores retain their witnessed
                # social function and prompt bindings.
                shared = component in {"prefix", "suffix"}
                if component == "prefix" and not same_act:
                    continue
                if meaning == "return_greeting" and any(
                    r.part_of_speech == "adjective" for s in form["slots"]
                    for r in DEFAULT_MORPHOLOGY.parse_word(s["surface"])
                ):
                    continue  # an attention call cannot ambiguously appraise the listener
                if component == "prefix" and form["slots"][0]["part"] == "conjunction" \
                        and not same_act and not (reaction == "self_report" and concern):
                    continue
                if (component == "prefix" and _COORDINATION_RELATIONS.get(
                    form["slots"][0]["lemma"]) == "contrast"
                    and not any(link.relation == "contrast" for link in structure.clause_links)):
                    continue
                if component == "prefix" and form["slots"][0]["part"] == "interjection" \
                        and not same_act:
                    continue
                condition_question = (reaction in {"self_report", "emotion"} and concern
                                      and meaning == "reciprocal_question"
                                      and any(s["lemma"] == "как" for s in form["slots"]))
                bound_question = reaction == "self_report" and meaning == "slot_question" and any(
                    s.get("agreement_role") == "topic" for s in form["slots"])
                greeting = reaction == "smalltalk" and meaning == "return_greeting"
                appraisal_value = lexical_valence(" ".join(s["surface"] for s in form["slots"]))
                positive_appraisal = (reaction == "self_report" and valence is not None
                                      and valence > 0 and meaning == "appraise"
                                      and appraisal_value is not None and appraisal_value > 0)
                empathy = reaction in {"self_report", "emotion"} \
                    and valence is not None and valence < 0 \
                    and meaning == "empathize"
                # Non-referential speech acts transfer by turn and tone. A
                # content binding must still resolve against this very prompt.
                unbound = not any(s["binding"] == "prompt" or s.get("agreement_role")
                                  for s in form["slots"])
                neutral = unbound and meaning == "acknowledge"
                limit = unbound and meaning == "express_limit" and reaction in {
                    "question", "request", "fragment", "evaluation",
                }
                reciprocal = unbound and meaning == "reciprocal_question" and reaction == "question"
                amusement = reaction == "amusement" and meaning == "mirror_amusement"
                hook = meaning == "slot_question" and any(
                    s.get("source_role") == "unfamiliar_word" for s in form["slots"])
                if not (shared or greeting or condition_question or bound_question
                        or empathy or positive_appraisal or neutral or limit or reciprocal or hook
                        or amusement or directed_retort
                        ) and (
                    not same_act or relevance < 0.08
                ):
                    continue
                if not unbound and form.get("prompt_negated", False) != any(
                    c.negated for c in exchange_analysis(prompt)[0].clauses
                ):
                    continue
                if ("formal" in form["register"]) != ("formal" in register):
                    continue
                if "formal" in form.get("response_register", ()) and "formal" not in register:
                    continue
                if "coarse" in form.get("response_register", ()) and not (
                    "coarse" in register or meaning == "retort"
                    or reaction == "self_report" and valence is not None and valence < 0
                    and (meaning == "empathize" or (form.get("stimulus_valence") or 0) < 0)
                ):
                    continue
                if reaction == "emotion" and concern and not shared and meaning not in {
                    "empathize", "reciprocal_question",
                } and not (same_act and meaning == "acknowledge"
                           and all(s["part"] == "interjection" for s in form["slots"])):
                    continue
                if (reaction == "emotion" and concern and not same_act
                        and component == "suffix") and (
                    "smile" in form.get("response_register", ())
                    or any(s["surface"] == "!" for s in form["slots"])
                ):
                    continue
                source_valence = lexical_valence(row["prompt"])
                if source_valence is None and form["meaning"] == "empathize":
                    source_valence = lexical_valence(row["answer"])
                if reaction == "self_report":
                    report_content = {t.lemma for t in structure.tokens
                                      if t.part_of_speech in {"verb", "noun", "adjective"}}
                    previous_content = {t.lemma for t in exchange_analysis(row["prompt"])[0].tokens
                                        if t.part_of_speech in {"verb", "noun", "adjective"}}
                    if not shared and meaning == "appraise" and (
                        not report_content & previous_content and not positive_appraisal
                        or valence is not None and valence < 0
                        or condition and valence is None
                        and appraisal_value is not None and appraisal_value > 0
                    ):
                        continue
                    if not shared and meaning == "acknowledge" and concern:
                        continue
                    if meaning == "empathize" and not (valence is not None and valence < 0):
                        continue
                    if meaning == "reciprocal_question" and not condition_question:
                        continue
                    if component == "prefix" and condition and valence is None \
                            and form["slots"][0]["part"] not in {
                        "particle", "conjunction",
                    }:
                        continue
                    if shared and form["component"] == "prefix" and valence is not None \
                            and valence < 0 and form["slots"][0]["part"] not in {
                                "particle", "conjunction",
                            } and not (
                                source_valence is not None and source_valence < 0):
                        continue
                    if component == "suffix" and (
                        concern
                    ) \
                            and ("smile" in form.get("response_register", ())
                                 or any(s["surface"] == "!" for s in form["slots"])):
                        continue
                # Bindability is structural and can be decided before choosing
                # the reaction meaning. An unavailable slot must not starve an
                # otherwise witnessed acknowledgment.
                if any(len(reaction_slot_tokens(structure, slot))
                       != 1 for slot in form["slots"]
                       if slot["binding"] == "prompt" or slot.get("agreement_role")):
                    continue
                key = json.dumps(form, sort_keys=True)
                rank = relevance
                if key not in grouped:
                    grouped[key] = [form, [], 0, rank, row["response_act"]]
                item = grouped[key]
                item[1].append(row["answer_event_id"])
                item[2] += 1
                item[3] = max(item[3], rank)
        if upgrades:
            self._exchange_revision += 1
            async with self._database.transaction() as connection:
                await _store_part_constructions(
                    connection, event.guild_id, constructions, event.utc_timestamp.isoformat(),
                )
                await connection.executemany(
                    "UPDATE language_observed_exchanges SET response_forms_json = ?, "
                    "topic_links_json = ? "
                    "WHERE answer_event_id = ?", upgrades,
                )
        syntax = {r["construction_id"]: tuple(json.loads(r["roles_json"])) for r in
                  await self._database.fetchall(
                      "SELECT construction_id, roles_json FROM language_productive_constructions "
                      "WHERE guild_id = ? AND predicate_class = 'dialogue_act' "
                      "AND confidence >= 0.4 AND utility > -0.75", (event.guild_id,),
                  )}
        result = tuple(
            ObservedDialogueForm(
                tuple(form["slots"]), form["clause_kind"], act, response_act,
                tuple(ids), count, min(1.0, relevance), reaction or act,
                "exchange-part:" + hashlib.sha256(
                    json.dumps(form, sort_keys=True).encode()).hexdigest()[:24],
                tuple(form.get("response_register", ())),
                (("empathize" if valence is not None and valence < 0 else "check_condition")
                 if reaction in {"self_report", "emotion"}
                 and form["meaning"] == "reciprocal_question" and concern
                 else form["meaning"]),
                form["component"],
                form["construction_id"], syntax[form["construction_id"]], valence,
                form.get("stimulus_valence"), condition,
                *self._rule_fields(form, rule_data, current_features),
            )
            for form, ids, count, relevance, response_act in sorted(
                grouped.values(), key=lambda item: (
                    -item[3], -item[2], json.dumps(item[0], sort_keys=True))
            ) if form["construction_id"] in syntax
        )
        weighted = []
        for form in result:
            if form.rule_id:
                support, weights = rule_support[form.rule_id]
                weight = min(1.0, 0.5 + 0.4 * support / (support + 2)
                             + sum(weights) / len(weights) - 0.8)
                weighted.append(replace(form, rule_weight=weight,
                                        activation=form.activation * weight / form.rule_weight,
                                        activation_source_ids=activation_sources))
            else:
                weighted.append(form)
        specialized = await self._computed_rules(event, specialized)
        forms = attach_reported_time(tuple(weighted) + specialized, event, prompt, rows)
        result = []
        for form in forms:
            if form.component == "core" and not form.mixture:
                bindings = tuple((s.get("source_role", "topic"), t.lemma, event.event_id)
                                 for s in form.slots if s["binding"] == "prompt"
                                 for t in reaction_slot_tokens(structure, s))
                strengths = {"known": form.activation * min(1.0, len(bindings) * 0.5),
                             "hypothesis": form.activation if form.hypothesis else 0.0,
                             "reaction": form.activation * (
                                 1 + abs(form.event_valence or 0.0) + 0.15 * len(register))}
                if not sum(strengths.values()):
                    strengths["reaction"] = form.activation or 1.0
                total = sum(strengths.values())
                form = replace(form, known_slots=tuple(dict.fromkeys(
                    (*form.known_slots, *bindings))),
                               mixture=tuple((k, v / total) for k, v in strengths.items() if v))
            result.append(replace(form, component="reserve") if form.recovery_only else form)
        if accepting_report or reaction == "acknowledgment":
            result = [f for f in result if f.component != "core" or f.meaning == "acknowledge"]
        return tuple(result)

    async def _computed_rules(self, event, forms):
        """Keep derived grammar rules on their original human source edges."""
        active = []
        states = {}
        for form in forms:
            if form.component != "core":
                if form.rule_id in states:
                    active.append(form)
                continue
            source = form.evidence_ids[0]
            owner = await self._database.fetchone(
                "SELECT answer_event_id,topic_links_json FROM language_observed_exchanges "
                "WHERE guild_id=? AND (answer_event_id=? OR prompt_event_id=?) "
                "ORDER BY observed_at DESC LIMIT 1", (event.guild_id, source, source),
            )
            if not owner:
                continue
            links = json.loads(owner["topic_links_json"])
            rule = next((r for r in links if r.get("rule_id") == form.rule_id), None)
            if rule is None:
                operator = "evaluation_tone" if form.meaning == "acknowledge" else "typed_question"
                rule = {"kind": "feature_transformation", "projection_operator": operator,
                        "rule_id": form.rule_id, "input": [f for f in form.input_features
                            if f.startswith(("turn:", "tone:", "valence:"))],
                        "target": list(form.target_features), "weight": form.rule_weight,
                        "support": form.support, "source_ids": list(form.evidence_ids),
                        "exceptions": [],
                        "feedback_ids": [], "feedback_targets": []}
                rule["added"], rule["removed"] = rule["target"], rule["input"]
                links.append(rule)
                self._exchange_revision += 1
                await self._database.execute(
                    "UPDATE language_observed_exchanges SET topic_links_json=? "
                    "WHERE answer_event_id=?",
                    (json.dumps(links, ensure_ascii=False), owner["answer_event_id"]),
                )
            if "|".join(sorted(form.input_features)) in rule["exceptions"]:
                continue
            if not set(rule["input"]) <= set(form.input_features):
                continue
            current = activate_transformation(rule, set(form.input_features))
            activation = min(current, form.activation * rule["weight"] / form.rule_weight)
            if activation <= 0:
                continue
            states[form.rule_id] = True
            active.append(replace(form, rule_weight=rule["weight"], support=rule["support"],
                                  activation=activation))
        return tuple(active)

    @staticmethod
    def _rule_fields(form, rules, features):
        rule, activation = rules.get(json.dumps(form, sort_keys=True), ({}, 0.0))
        return (rule.get("rule_id"), rule.get("weight", 0.0), activation,
                tuple(sorted(features)), tuple(rule.get("target", ())),
                tuple(rule.get("exceptions", ())))


@lru_cache(maxsize=16384)
def cached_response_forms(prompt: str, answer: str):
    text = json.dumps([prompt, answer], ensure_ascii=False)
    payload = _ANALYSIS_PAYLOADS.get(("forms", text))
    if payload is not None:
        return _decode_analysis(payload)
    return response_forms(prompt, answer)


def projected_slot_features(structure, slot, token=None):
    """Project tense from the report; a donor's present tense is not a rule."""
    features = {k: v for k, v in slot["features"].items()
                if k in {"case", "number", "gender", "person", "tense"}}
    if slot["binding"] in {"speech_act", "punctuation"}:
        return {}
    if token is None:
        return features
    current = dict(token.grammatical_features)
    if slot.get("agreement_role"):
        features.update({k: v for k, v in current.items() if k in {"case", "number", "gender"}})
    elif slot["part"] == "noun":
        features.update({k: v for k, v in current.items() if k in {"number", "gender"}})
    elif slot["part"] == "verb" and features.get("tense"):
        main = next((structure.tokens[c.predicate_index] for c in structure.clauses
                     if c.predicate_index is not None), token)
        finite = dict(main.grammatical_features)
        tense = current.get("tense") or finite.get("tense")
        if tense:
            features["tense"] = tense
        features["number"] = current.get("number", finite.get("number", "singular"))
        if tense == "past":
            features.pop("person", None)
            if current.get("gender") or finite.get("gender"):
                features["gender"] = current.get("gender", finite.get("gender"))
        else:
            features.pop("gender", None)
    if slot.get("source_role") in {"unfamiliar_word", "heard_surface"}:
        return {}
    return features


async def own_turn_history(database, guild_id: int, actor_id: int, before, limit: int = 64):
    """The agent's own delivered turns to one person on one server, newest first, no window.

    Sources: delivered meaning intentions (any channel of the server) and, for turns sent
    before intentions recorded a winning guess, the receipts kept by sleep hypotheses of
    the same person. Other people and other servers are never read. Each item carries the
    position taken (``claim``: a statement the turn asserted, else ``None``).
    """
    def basis_of(row):
        return (json.loads(row["selection_json"]) or {}).get("basis")

    rows = await database.fetchall(
        "SELECT intention_id,channel_id,response_message_id,sent_at,response_text,plan_json,"
        "selection_json FROM language_meaning_intentions WHERE guild_id=? AND user_id=? "
        "AND response_message_id IS NOT NULL AND sent_at<? ORDER BY sent_at DESC LIMIT ?",
        (guild_id, actor_id, before.isoformat(), limit))
    items, seen = [], set()
    for row in rows:
        plan = json.loads(row["plan_json"])
        claim = plan.get("claim") or {}
        seen.add(row["response_message_id"])
        items.append({
            "sent_at": row["sent_at"], "intention_id": row["intention_id"],
            "receipt_id": f"receipt:{row['channel_id']}:{row['response_message_id']}",
            "text": row["response_text"] or "",
            "claim": claim if claim.get("clause_kind") == "statement"
            and not plan.get("knowledge_limit")
            and not (basis_of(row) or {}).get("fantasy") else None,
            "basis": basis_of(row) or {}})
    hypotheses = await database.fetchall(
        "SELECT hypothesis_id,roles_json FROM continuous_cognition_hypotheses "
        "WHERE guild_id=? AND predicate_key='sleep_response'", (guild_id,))
    for row in hypotheses:
        data = json.loads(row["roles_json"])
        if data.get("actor_id") != actor_id:
            continue
        claim = ((data.get("candidate") or {}).get("metadata") or {}).get(
            "message_plan", {}).get("claim")
        for receipt in data.get("receipts", ()):
            if receipt in seen:
                continue
            sent = await database.fetchone(
                "SELECT occurred_at,payload_json FROM events WHERE event_id=? AND guild_id=?",
                (f"discord:message:{receipt}", guild_id))
            if sent is None or sent["occurred_at"] >= before.isoformat():
                continue
            seen.add(receipt)
            items.append({
                "sent_at": sent["occurred_at"], "intention_id": row["hypothesis_id"],
                "receipt_id": f"receipt:{data['channel_id']}:{receipt}",
                "text": str(json.loads(sent["payload_json"]).get("content") or ""),
                "claim": claim if claim and claim.get("clause_kind") == "statement" else None,
                "basis": {}})
    items.sort(key=lambda item: item["sent_at"], reverse=True)
    return items[:limit]


def quoted_utterance_dependency(text: str):
    from debilgi.language.productive_nlg import ConceptArgument, ConceptLexeme, SemanticDependency

    quote = "«" + text.strip() + "»"
    return SemanticDependency("direct_object", ConceptArgument(
        ConceptLexeme(quote, quote, "noun", (("indeclinable", "true"),),
                      entity_kind="quoted_utterance")))


_VERB_ENDINGS = ("ться", "ить", "ять", "ать", "еть", "уть", "ыть", "оть", "ть", "ти", "чь")


def _same_root_pair(imperfective: str, perfective: str) -> bool:
    """A strict aspect pair: the perfective is a prefix + the imperfective («сделать»/«делать»),
    or one stem extends the other («отправить»/«отправлять», «изучить»/«изучать»).

    The morphological pair reader is permissive («обращать» pairs with «обработать»); a habit
    stated with a wrong verb would be false, so only a pair that is clearly one root is used.
    """
    from debilgi.language.lexical_matching import _VERBAL_PREFIXES
    from debilgi.language.morphology import DEFAULT_MORPHOLOGY

    def transitivity(word):
        return {r.transitivity for r in DEFAULT_MORPHOLOGY.parse_word(word)
                if r.is_known and r.part_of_speech == "verb"}

    if transitivity(imperfective) and transitivity(perfective) and not (
            transitivity(imperfective) & transitivity(perfective)):
        return False  # «работать» is not the imperfective of «обработать»
    if any(perfective == prefix + imperfective for prefix in _VERBAL_PREFIXES):
        return True

    def stem(word):
        return next((word[:-len(e)] for e in _VERB_ENDINGS if word.endswith(e)), word)

    short, long = stem(perfective), stem(imperfective)
    return len(short) >= 4 and long.startswith(short)


def delivered_report_plan(goal, turn: dict):
    """«Я сказал «…».»: the report of one actually delivered turn, with its receipt."""
    from debilgi.config import thought_content_enabled
    from debilgi.language.communication import MessagePlan
    from debilgi.language.generative_planner import DerivedPlan, SemanticEvidence
    from debilgi.language.productive_nlg import (
        ClauseThought,
        ConceptArgument,
        ConceptLexeme,
        SemanticDependency,
    )

    evidence = (turn["receipt_id"],)
    subject = SemanticDependency("subject", ConceptArgument(ConceptLexeme(
        "я", "я", "pronoun", (("number", "singular"), ("person", "first")))))
    quoted = turn["text"]
    if thought_content_enabled():
        # The quotation marks hold the words themselves: a report of a report is unwrapped to the
        # original words (no «Я сказал «Я сказал …»»), and the sentence mark of the quoted turn is
        # not carried inside (it closed that turn, not this one).
        if "«" in quoted and "»" in quoted:
            quoted = quoted[quoted.index("«") + 1:quoted.rindex("»")]
        quoted = quoted.strip().rstrip(".!?…").rstrip()
    thought = ClauseThought(
        ConceptLexeme("сказать", "сказать", "verb", (
            ("tense", "past"), ("number", "singular"), ("gender", "masculine"))),
        (subject, quoted_utterance_dependency(quoted)),
        confidence=1.0, evidence=evidence)
    plan = replace(MessagePlan(goal, thought, mixture=(("known", 1.0),)),
                   known_slots=(("speech_act", turn["text"], evidence[0]),))
    return DerivedPlan(plan, "verify_reported_position", (
        SemanticEvidence(evidence[0], "sent_turn", turn["text"], plan.claim),))


def own_turn_plan(context, decision):
    """Explain an actual delivered speech act, not an invented personal history."""
    from debilgi.language.canonical_claims import ClaimQuery
    from debilgi.language.communication import KnowledgeLimit, MessagePlan, reader_goal
    from debilgi.language.dialogue_obligations import thought_from_data
    from debilgi.language.generative_planner import DerivedPlan, SemanticEvidence
    from debilgi.language.productive_nlg import (
        ClauseThought,
        ConceptArgument,
        ConceptLexeme,
        SemanticDependency,
    )

    prior = context.prior_bot_turn
    data = prior["plan"]
    evidence = (prior.get("receipt_id", prior["prompt_event_id"]),)
    goal = reader_goal(context.structure)
    limit = data.get("knowledge_limit")
    if limit and getattr(context, "focus_reason", None) == "answer_control_keeps_request":
        # «Ну так что? Ответишь?» asks again for the focused request. The delivered turn
        # is its explanation only if that turn was the answer to the same request.
        from debilgi.language.lexical_matching import aspect_pair

        asked = getattr(getattr(context, "claim_result", None), "query", None)
        said = (limit.get("query") or {}).get("predicate")
        if (asked is not None and asked.predicate and said
                and asked.predicate != said and not aspect_pair(asked.predicate, said)):
            return None
    if limit:
        thought = thought_from_data(data["claim"])
        operation = prior["operations"].get("epistemic")
        if operation and operation[0] == thought.predicate.lemma:
            # The retelling keeps the verb that was actually said; a different learned
            # epistemic operation is not what the delivered limit used.
            evidence += (operation[1],)
        query = ClaimQuery(**{**limit["query"],
                              "arguments": tuple(map(tuple, limit["query"].get("arguments", ())))})
        plan = MessagePlan(goal, replace(thought, evidence=evidence),
                           knowledge_limit=KnowledgeLimit(
            query, thought_from_data(limit["question"]) if limit.get("question") else None,
            limit.get("question_word")),
            mixture=(("known", 1.0),))
    else:
        greeting_operation = prior["operations"].get("greeting")
        meaning = (prior["decision"].get("target_meaning")
                   or data.get("claim", {}).get("predicate", {}).get("lemma"))
        greeting = (meaning == "return_greeting"
                    or not meaning and prior["decision"].get("act") in {"greeting", "call"}
                    or greeting_operation and meaning == greeting_operation[0]
                    or data.get("claim", {}).get("predicate_class") == "dialogue_act"
                    and "greeting" in analyze_pragmatics(prior["response_text"]).functions)
        operation = prior["operations"].get("greeting") if greeting else None
        dependencies = (SemanticDependency("subject", ConceptArgument(
            ConceptLexeme("я", "я", "pronoun", (("number", "singular"), ("person", "first"))))),)
        if operation:
            predicate = operation[0]
            evidence += (operation[1],)
            dependencies += (SemanticDependency("direct_object", ConceptArgument(
                ConceptLexeme("ты", "ты", "pronoun"), grammatical_case="accusative")),)
        else:
            predicate = "сказать"  # existing speech-report operator, with an actual receipt
            dependencies += (quoted_utterance_dependency(prior["response_text"]),)
        thought = ClauseThought(ConceptLexeme(predicate, predicate, "verb", (
            ("tense", "past"), ("number", "singular"), ("gender", "masculine"))),
            dependencies, confidence=1.0, evidence=evidence)
        # Another repair rephrases the same delivered report. It must not
        # recursively quote its own explanation on every conversational turn.
        previous_claim = data.get("claim", {})
        if (not operation and previous_claim.get("predicate", {}).get("lemma") == predicate
                and any(d.get("argument", {}).get("head", {}).get("entity_kind")
                        == "quoted_utterance" for d in previous_claim.get("dependencies", ()))):
            thought = replace(thought_from_data(previous_claim), evidence=evidence)
        plan = MessagePlan(goal, thought, mixture=(("known", 1.0),))
    plan = replace(plan, known_slots=(("speech_act", prior["response_text"], evidence[0]),))
    return DerivedPlan(plan, "retrieve_delivered_act", (
        SemanticEvidence(evidence[0], "sent_turn", prior["response_text"], plan.claim),
    ), "почему" if goal.question_word == "почему" else None)


def reverse_reaction_check(plan, parts, prompt: str, text: str) -> dict:
    """Reparse the produced utterance, not a reconstruction of its source parts."""
    core = next(p for p in parts if p.component == "core")
    structure = exchange_analysis(text)[0]
    projected = _reaction_parts(prompt, text)
    meanings = {p["meaning"] for p in projected if p["component"] == "core"}
    target = plan.claim.predicate.lemma
    if core.hypothesis and any(s["binding"] == "hypothesis" for s in core.slots):
        meanings = {target} if question_surface_preserved(core, prompt, text) else set()
    if target == "perform_contact":
        from debilgi.cognition.dialogue_tasks import contact_response_binding

        meanings = {target} if contact_response_binding(text) >= 0.8 else set()
    # A condition inquiry has the same interrogative grammar as its raw
    # reciprocal observation; event valence supplies its contextual function.
    compatible = {target}
    if target in {"empathize", "check_condition"} and core.clause_kind == "question":
        compatible.add("reciprocal_question")
    issues = []
    if (dict(core.hypothesis or plan.hypothesis).get("confirmation") == "polar"
            and any(c.question_target_index is not None for c in structure.clauses)):
        issues.append("hypothesis_confirmation_domain_changed")
    if not meanings & compatible:
        issues.append("target_meaning_mismatch")
    question = any(c.clause_kind == "question" for c in structure.clauses) or "?" in text
    if question != (core.clause_kind == "question"):
        issues.append("target_clause_mismatch")
    source = exchange_analysis(prompt)[0]
    if (plan.goal.addressed and not plan.goal.question_word
            and any(c.clause_kind == "question" for c in source.clauses)
            and any(s["part"] == "verb" and s["binding"] == "prompt" for s in core.slots)
            and not any(s.get("surface") == "?" for s in core.slots)
            and not any(
                s["part"] in {"pronoun", "adverb"} and any(
                    {"NPRO", "ADVB"} & r.grammemes
                    and not {"1per", "2per", "3per"} & r.grammemes
                    for r in DEFAULT_MORPHOLOGY.parse_word(s["surface"]))
                for s in core.slots)
            and not any(c.question_target_index is not None for c in structure.clauses)):
        # The verb of a polar question put to the agent, restated without a question
        # mark or a question word, asserts what the agent itself does or wants. No one
        # has confirmed it, and a reaction carries no hypothesis marker.
        issues.append("unsupported_self_assertion")
    if unbound_anaphors(structure, source):
        # Same test as at learning time, on the words actually produced for this turn:
        # a borrowed «кто их не V?» must not outlive the noun it pointed at.
        issues.append("unbound_anaphor")
    for slot in core.slots:
        if slot["binding"] == "hypothesis":
            expected = projected_slot_features(source, slot, next((
                source.tokens[c.predicate_index] for c in source.clauses
                if c.predicate_index is not None), None))
            output = [t for t in structure.tokens if t.lemma == slot["lemma"]]
            if not output or not any(all(r.features().get(k) == v for k, v in expected.items())
                                    for t in output
                                    for r in DEFAULT_MORPHOLOGY.parse_word(t.surface)):
                issues.append("hypothesis_morphology_mismatch")
        if slot["binding"] == "prompt" and slot.get("source_role") != "unfamiliar_word":
            bound = reaction_slot_tokens(source, slot)
            if len(bound) == 1:
                output = [t for t in structure.tokens if t.lemma == bound[0].lemma]
                if not output:
                    issues.append("missing_bound_content")
                elif slot["part"] == "verb":
                    expected = {k: v for k, v in projected_slot_features(
                        source, slot, bound[0]).items()
                                if k in {"person", "tense"}}
                    expected.update((k, v) for k, v in bound[0].grammatical_features
                                    if k == "number")
                    if not any(all(r.features().get(k) == v for k, v in expected.items())
                               for t in output for r in DEFAULT_MORPHOLOGY.parse_word(t.surface)):
                        issues.append("predicate_morphology_mismatch")
        if slot.get("agreement_role"):
            bound = reaction_slot_tokens(source, slot)
            output = [t for t in structure.tokens if t.lemma == slot["lemma"]]
            if len(bound) == 1 and output:
                expected = dict(bound[0].grammatical_features)
                agreement = {k: v for k, v in expected.items()
                             if k in {"number", "gender", "case"}
                             and not (k == "gender" and expected.get("number") == "plural")}
                if not any(all(r.features().get(k) == v for k, v in agreement.items())
                           for r in DEFAULT_MORPHOLOGY.parse_word(output[0].surface)):
                    issues.append("agreement_mismatch")
    return {"passed": not issues, "target": target, "observed": sorted(meanings),
            "question": question, "issues": sorted(set(issues)),
            "features": sorted(turn_features(text))}


def observed_reaction_binding(prompt: str, text: str, metadata) -> float:
    """Reproduce the selected meaning and its parts through the common composer."""
    from debilgi.language.communication import MessagePlan, ReaderGoal, ResponseComposer
    from debilgi.language.dialogue_obligations import thought_from_data

    decision, data = metadata.get("response_decision"), metadata.get("message_plan")
    if not isinstance(decision, dict) or not isinstance(data, dict):
        return 0.0
    parts = metadata.get("reaction_parts")
    if (decision.get("mode") != "free_form" or not parts
            or decision.get("act") != data.get("speech_act") or metadata.get("hard_rejection")
            or not metadata.get("reader_checked") or not metadata.get("thought_before_surface")):
        return 0.0
    forms = tuple(ObservedDialogueForm(**part) for part in parts)
    cores = [f for f in forms if f.component == "core"]
    if len(cores) != 1 or (cores[0].clause_kind == "question"
                           and not decision.get("allow_hook_question")):
        return 0.0
    plan = MessagePlan(ReaderGoal(**data["goal"]), thought_from_data(data["claim"]),
                       speech_act=data["speech_act"])
    if plan.claim.predicate_class != "dialogue_act" or not plan.claim.evidence:
        return 0.0
    surface = ResponseComposer.reaction_surface(plan, forms, prompt)
    return float(surface is not None and surface.text == text)


def realize_observed_form(form: ObservedDialogueForm, prompt: str) -> str | None:
    """Bind only current-turn content; corpus-open slots require other evidence.

    Statements with a first-person participant cannot be inferred from an
    outsider's self report. Unbound content causes abstention, never a sampled
    dog, occupation, family member, or evaluative state.
    """
    structure, _, _ = exchange_analysis(prompt)
    if form.reactive_act:
        return None  # reaction parts are realized only through ResponseComposer
    if (
        form.clause_kind != "question"
        and any(c.clause_kind in {"question", "command"} for c in structure.clauses)
        and not all(s["binding"] == "speech_act" for s in form.slots)
    ):
        return None
    if form.clause_kind != "question" and (
        any(s.get("lemma") == "не" for s in form.slots) != any(c.negated for c in structure.clauses)
    ):
        return None
    if (
        form.clause_kind != "question"
        and not any(s["binding"] == "prompt" for s in form.slots)
        and not all(s["binding"] == "speech_act" for s in form.slots)
    ):
        return None
    if form.clause_kind != "question" and any(
        s.get("lemma") in {"я", "мы", "мой", "наш"} for s in form.slots
    ):
        return None
    result = []
    for slot in form.slots:
        binding = slot["binding"]
        if binding == "open":
            return None
        if binding in {"grammar", "speech_act"}:
            result.append(slot.get("surface", slot["lemma"]))
            continue
        options = [
            t
            for t in structure.tokens
            if t.part_of_speech == slot["part"] and t.semantic_role == slot["source_role"]
        ]
        if len(options) != 1:
            return None
        token = options[0]
        features = dict(slot["features"])
        original_features = dict(token.grammatical_features)
        if original_features.get("number"):
            features["number"] = original_features["number"]
        if form.clause_kind != "question" and original_features.get("tense"):
            features["tense"] = original_features["tense"]
        # Morphology comes from the observed slot, lexical content from this turn.
        surface = DEFAULT_MORPHOLOGY.inflect(
            token.lemma,
            **{
                key: value
                for key, value in features.items()
                if key in {"case", "number", "person", "tense"}
            },
        )
        if surface is None:
            return None
        result.append(surface)
    if not result:
        return None
    text = " ".join(result)
    return text[:1].upper() + text[1:] + ("?" if form.clause_kind == "question" else ".")


@lru_cache(maxsize=16384)
def _question_units(source_id, text):
    """Human question clauses projected to grammar and typed content variables."""
    payload = _ANALYSIS_PAYLOADS.get(("units", text))
    if payload is not None:
        return tuple(dict(unit, source_id=source_id)
                     for unit in _decode_analysis(payload))
    from debilgi.language.canonical_claims import RELATIVE_DAYS

    structure = exchange_analysis(text)[0]
    units = []
    for clause in structure.clauses:
        if clause.clause_kind != "question":
            continue
        tokens = [t for t in structure.tokens[clause.token_start:clause.token_end]
                  if t.surface[:1].isalnum()]
        if not tokens or len(tokens) > 7 or any(
            t.part_of_speech == "pronoun" and t.lemma not in {"ты", "вы", "что", "какой"}
            for t in tokens
        ):
            continue
        verbs = [t for t in tokens if t.part_of_speech == "verb"]
        nouns = [t for t in tokens if t.part_of_speech == "noun"]
        states = [t for t in tokens if t.part_of_speech in {"adverb", "predicate"}
                  and t.semantic_role != "question_target" and any(
                      "Prdx" in r.grammemes or r.part_of_speech == "predicate"
                      for r in DEFAULT_MORPHOLOGY.parse_word(t.surface))]
        if (len(verbs) > 1 or len(nouns) > 1
                or sum(t.lemma == "не" for t in tokens) > 1):
            continue
        pivot = verbs[0] if verbs else states[0] if len(states) == 1 else (
            nouns[0] if len(nouns) == 1 else None)
        if pivot is None or verbs and nouns:
            continue
        wh = next((t.lemma for t in tokens if t.semantic_role == "question_target"), None)
        if nouns and not verbs and wh not in {"что", "какой"}:
            continue
        slots = []
        for token in tokens:
            slot = {"lemma": token.lemma, "surface": token.surface,
                    "part": token.part_of_speech, "features": dict(token.grammatical_features)}
            if token.index == pivot.index:
                slot.update(binding="prompt", source_role=(
                    "predicate" if verbs else "condition_word" if states else "topic"))
            elif token.lemma in RELATIVE_DAYS:
                slot.update(binding="prompt", source_role="deictic_time")
            elif token.part_of_speech in _CLOSED | {"adverb", "interjection"} \
                    or token.semantic_role == "question_target":
                slot["binding"] = "grammar"
                if token.part_of_speech == "adjective" and nouns:
                    slot["agreement_role"] = "topic"
            else:
                break
            slots.append(slot)
        else:
            units.append({"source_id": source_id, "slots": slots, "pivot": pivot.lemma,
                          "part": pivot.part_of_speech, "wh": wh,
                          "kind": "verb" if verbs else "state" if states else "noun",
                          "infinitive": "INFN" in pivot.grammemes,
                          "negative": clause.negated,
                          "temporal": any(t.lemma in RELATIVE_DAYS for t in tokens)})
    return tuple(units)


def _computed_form(event, prompt, unit, *, meaning, activation, hypothesis=(), pivot_lemma=None):
    """Unify an observed grammar with current slots or one explicitly labelled analogy."""
    from debilgi.language.canonical_claims import RELATIVE_DAYS
    from debilgi.language.morphology import DEIXIS_CELLS, person_cell

    structure = exchange_analysis(prompt)[0]
    source = unit["source_id"]
    slots = [dict(s, features=dict(s["features"])) for s in unit["slots"]]
    known = []
    for slot in slots:
        if slot["binding"] == "prompt" and slot["lemma"] == unit["pivot"]:
            if pivot_lemma:
                slot.update(binding="hypothesis", lemma=pivot_lemma, surface=pivot_lemma)
            elif unit["kind"] == "verb" and unit.get("request"):
                slot["source_role"] = "requested_predicate"
            elif unit["kind"] == "state":
                slot["source_role"] = "condition_word"
        if (slot["binding"] == "grammar" and slot["part"] == "pronoun"
                and person_cell(slot["lemma"]) == ("second", "plural", "personal")):
            # One interlocutor is addressed: the plural second person turns singular.
            singular = DEIXIS_CELLS[("second", "singular")][0]
            slot.update(lemma=singular, surface=singular)
            slot["features"]["number"] = "singular"
        if slot["binding"] == "prompt" and slot.get("source_role") == "deictic_time":
            matches = [t for t in structure.tokens if t.lemma in RELATIVE_DAYS]
            if not matches:
                return ()
        if slot.get("source_role") == "topic":
            known.extend(("topic", t.lemma, event.event_id) for t in
                         reaction_slot_tokens(structure, slot))
    # Append a reported temporal argument only with a human temporal-order witness.
    time = next((t for t in structure.tokens if t.lemma in RELATIVE_DAYS), None)
    if time and not any(s.get("source_role") == "deictic_time" for s in slots):
        witness = unit.get("time_witness")
        if witness:
            temporal = {"binding": "prompt", "source_role": "deictic_time", "part": "adverb",
                        "lemma": time.lemma, "surface": time.surface, "features": {}}
            slots.insert(len(slots) if witness[1] == "after" else 0, temporal)
    if time and any(s.get("source_role") == "deictic_time" for s in slots):
        known.append(("time", time.lemma, event.event_id))
    if unit["kind"] in {"state", "verb"} and not pivot_lemma:
        known.append(("predicate", unit["pivot"], event.event_id))
    roles = tuple(f"act:{i}" for i in range(len(slots)))
    identity = json.dumps((unit["kind"], unit["slots"], meaning,
                           "analogy" if pivot_lemma else "bind", sorted(
                               f for f in turn_features(prompt)
                               if f.startswith(("turn:", "tone:", "valence:")))), sort_keys=True)
    rule_id = "exchange-rule:" + hashlib.sha256(identity.encode()).hexdigest()[:24]
    syntax = "exchange-syntax:" + hashlib.sha256(identity.encode()).hexdigest()[:24]
    hyp = dict(hypothesis)
    if hyp:
        hyp.update(rule_id=rule_id, source_event_id=event.event_id)
        if hyp.get("predicate"):
            from debilgi.language.canonical_claims import semantic_time
            from debilgi.language.productive_nlg import (
                ClauseThought,
                ConceptArgument,
                ConceptLexeme,
                SemanticDependency,
            )

            clause = next(iter(event.meaning.clauses), None)
            tense = clause.semantic_time.tense if clause else None
            predicate = ConceptLexeme(hyp["predicate"], hyp["predicate"], "verb",
                                      (("tense", tense or "past"),))
            subject = ConceptLexeme("ты", "ты", "pronoun", (("number", "singular"),),
                                    participant_ref=f"discord:user:{event.actor_id}")
            proposition = ClauseThought(predicate, (SemanticDependency(
                "subject", ConceptArgument(subject)),), confidence=activation,
                semantic_time=semantic_time(clause.semantic_time, event, None) if clause else None)
            hyp["proposition"] = json.dumps(identity_graph(proposition), ensure_ascii=False)
            hyp["confirmation"] = "polar" if not unit["wh"] and not unit["negative"] else "none"
    strengths = {"known": activation * min(1.0, len(known) * 0.5),
                 "hypothesis": activation if hyp else 0.0,
                 "reaction": activation * abs(lexical_valence(prompt) or 0.0) * 0.5}
    if not sum(strengths.values()):
        strengths["reaction"] = activation
    total = sum(strengths.values())
    mixture = tuple((k, v / total) for k, v in strengths.items() if v)
    evidence = tuple(dict.fromkeys((source, *unit.get("graph_sources", ()),
                                   *((unit["time_witness"][0],)
                                     if unit.get("time_witness") else ()))))
    core = ObservedDialogueForm(
        tuple(slots), "question", analyze_turn_form(prompt).kind, "question", evidence, 1,
        activation, reactive_act(prompt) or analyze_turn_form(prompt).kind,
        "exchange-part:" + hashlib.sha256(identity.encode()).hexdigest()[:24], (), meaning, "core",
        syntax, roles, lexical_valence(prompt), None, False, rule_id, 0.7, activation,
        tuple(sorted(turn_features(prompt))), ("meaning:" + meaning, "clause:question"), (),
        unit.get("graph_sources", ()), tuple(sorted(hyp.items())), tuple(known), mixture,
    )
    mark = {"binding": "punctuation", "part": None, "surface": "?", "lemma": "?", "features": {}}
    suffix = replace(core, slots=(mark,), component="suffix", meaning="tone",
                     construction_roles=("act:0",), pattern_id=core.pattern_id + ":mark",
                     hypothesis=(), mixture=(), known_slots=())
    return core, suffix


def computed_question_forms(event, prompt, rows):
    """Bounded graph analogies and reference questions, through the same form/composer path."""
    from debilgi.language.canonical_claims import RELATIVE_DAYS
    from debilgi.language.lexical_matching import predicate_family

    structure = exchange_analysis(prompt)[0]
    reaction = reactive_act(prompt) or analyze_turn_form(prompt).kind
    noun_tokens = [t for t in structure.tokens if t.part_of_speech == "noun"
                   and t.semantic_role not in {"time", "duration", "pragmatic_marker"}]
    verb_tokens = [t for t in structure.tokens if t.part_of_speech == "verb"]
    bare_topic = (len(noun_tokens) == 1 and not verb_tokens
                  and reaction in {"fragment", "question"} and len(structure.tokens) <= 5)
    condition = (reaction in {"self_report", "emotion"} and not any(
        c.negated or c.direct_object_indices or c.action_complement_indices
        for c in structure.clauses))
    request = reaction == "request" and bool(verb_tokens)
    if not (bare_topic or condition or request):
        return computed_contact_form(event, prompt, rows)
    if not isinstance(rows, _ExchangeRows):
        projection = _ExchangeRows()
        projection.update(rows)
        rows = projection
    units, temporal, all_neighbours = rows.question_projection()
    topics = {t.lemma for t in noun_tokens}
    neighbours = {}
    for topic in topics:
        for verb, sources in all_neighbours.get(topic, {}).items():
            neighbours.setdefault(verb, set()).update(sources)
    main = next((structure.tokens[c.predicate_index] for c in structure.clauses
                 if c.predicate_index is not None), None)
    choices = []
    for raw in units:
        unit = dict(raw, slots=[dict(slot, features=dict(slot["features"]))
                                for slot in raw["slots"]], time_witness=temporal)
        if bare_topic and unit["kind"] == "noun":
            if any(s["part"] == "pronoun" and s["lemma"] not in {"что", "это"}
                   for s in unit["slots"]):
                continue
            if unit["wh"] == "что" and any(s["lemma"] == "за" for s in unit["slots"]):
                for s in unit["slots"]:
                    if s.get("source_role") == "topic":
                        s["features"]["case"] = "nominative"
            choices.append((0.32, _computed_form(event, prompt, unit, meaning="slot_question",
                activation=0.32, hypothesis=(("kind", "reference"),
                                            ("topic", noun_tokens[0].lemma)))))
        elif reaction == "emotion" and unit["kind"] == "state" and any(
            t.lemma == unit["pivot"] for t in structure.tokens
        ):
            choices.append((0.38, _computed_form(event, prompt, unit, meaning="slot_question",
                                               activation=0.38)))
        elif (condition and main and main.lemma != "быть" and unit["kind"] == "verb"
                and not unit["infinitive"] and predicate_family(main.lemma, unit["pivot"])
                and (main.lemma == unit["pivot"] or not noun_tokens)):
            if not unit["wh"] and not any(
                s["part"] == "adverb" and s["lemma"] not in RELATIVE_DAYS
                for s in unit["slots"]
            ):
                continue  # do not re-ask (or negate) the event just reported
            score = 0.38 if main.lemma == unit["pivot"] else 0.34
            # A negative degree question has evidence of concern; no event-name lexicon.
            score += 0.015 if unit["negative"] else 0.0
            choices.append((score, _computed_form(event, prompt, unit, meaning="slot_question",
                activation=score, hypothesis=(("kind", "predicate_family"),
                                             ("predicate", main.lemma)), pivot_lemma=main.lemma)))
        elif condition and main and main.lemma == "быть" and unit["kind"] == "verb" \
                and not unit["infinitive"] and unit["pivot"] in neighbours:
            count = len(neighbours[unit["pivot"]])
            score = 0.24 + 0.12 * count / (count + 2) + (0.02 if unit["wh"] is None else 0.0)
            unit["graph_sources"] = tuple(sorted(neighbours[unit["pivot"]]))[:12]
            choices.append((score, _computed_form(event, prompt, unit, meaning="slot_question",
                activation=score, hypothesis=(("kind", "predicate_analogy"),
                    ("predicate", unit["pivot"]), ("topic", next(iter(topics), ""))),
                pivot_lemma=unit["pivot"])))
        elif request and unit["kind"] == "verb" and unit["infinitive"] and unit["wh"] == "что":
            unit["request"] = True
            choices.append((0.33, _computed_form(event, prompt, unit, meaning="slot_question",
                activation=0.33, hypothesis=(("kind", "requested_argument"),))))
    choices.sort(key=lambda x: (-x[0], len(x[1][0].slots) if x[1] else 100,
                                x[1][0].pattern_id if x[1] else ""))
    # One target predicate and at most three grammatical realizations; no recursive hypotheses.
    selected, key = [], None
    for _, forms in choices:
        if not forms:
            continue
        meaning = dict(forms[0].hypothesis).get("predicate") or forms[0].meaning
        if key is None:
            key = meaning
        if meaning == key:
            selected.extend(forms)
        if len(selected) >= 6:
            break
    return tuple(selected)


def learned_time_edge(rows):
    from debilgi.language.canonical_claims import RELATIVE_DAYS

    if isinstance(rows, _ExchangeRows) and rows._time_ready:
        return rows._time_edge
    before = None
    for row in rows:
        for source_id, text in ((row["answer_event_id"], row["answer"]),
                                (row["prompt_event_id"], row["prompt"])):
            structure = exchange_analysis(text)[0]
            for clause in structure.clauses:
                if clause.clause_kind != "question" or clause.predicate_index is None:
                    continue
                for t in structure.tokens[clause.token_start:clause.token_end]:
                    if t.lemma in RELATIVE_DAYS:
                        if t.index > clause.predicate_index:
                            if isinstance(rows, _ExchangeRows):
                                rows._time_edge, rows._time_ready = (source_id, "after"), True
                            return source_id, "after"
                        before = before or (source_id, "before")
    if isinstance(rows, _ExchangeRows):
        rows._time_edge, rows._time_ready = before, True
    return before


def attach_reported_time(forms, event, prompt, rows):
    from debilgi.language.canonical_claims import RELATIVE_DAYS

    structure = exchange_analysis(prompt)[0]
    time = next((t for t in structure.tokens if t.lemma in RELATIVE_DAYS), None)
    witness = learned_time_edge(rows) if time else None
    if not witness:
        return forms
    result = []
    for form in forms:
        if (form.component == "core" and form.clause_kind == "question"
                and any(s["part"] == "verb" for s in form.slots)
                and not any(s.get("source_role") == "deictic_time" for s in form.slots)):
            temporal = {"binding": "prompt", "source_role": "deictic_time", "part": "adverb",
                        "lemma": time.lemma, "surface": time.surface, "features": {}}
            slots = list(form.slots)
            slots.insert(len(slots) if witness[1] == "after" else 0, temporal)
            form = replace(form, slots=tuple(slots),
                           construction_roles=tuple(f"act:{i}" for i in range(len(slots))),
                           construction_id=form.construction_id + ":time",
                           pattern_id=form.pattern_id + ":time",
                           evidence_ids=tuple(dict.fromkeys((*form.evidence_ids, witness[0]))),
                           known_slots=(*form.known_slots, ("time", time.lemma, event.event_id)))
        result.append(form)
    return tuple(result)


def question_surface_preserved(core, prompt, text):
    """A speculative predicate is licensed only inside one parsed question."""
    structure = exchange_analysis(text)[0]
    if len(structure.clauses) != 1 or not text.rstrip().endswith("?"):
        return False
    if structure.clauses[0].clause_kind != "question":
        return False
    allowed = {s["lemma"] for s in core.slots}
    source = exchange_analysis(prompt)[0]
    allowed.update(t.lemma for t in source.tokens)
    return all(t.lemma in allowed or not t.surface[:1].isalnum() for t in structure.tokens)


def computed_contact_form(event, prompt, rows):
    from debilgi.cognition.dialogue_tasks import _CONTACT_RESPONSE_ACTIONS
    from debilgi.language.lexical_matching import same_lexical_family

    structure = exchange_analysis(prompt)[0]
    verb = next((t for t in structure.tokens if t.part_of_speech == "verb"
                 and dict(t.grammatical_features).get("person") == "second"
                 and dict(t.grammatical_features).get("tense") == "future"
                 and t.lemma in _CONTACT_RESPONSE_ACTIONS), None)
    if not verb or any(t.part_of_speech == "noun" for t in structure.tokens):
        return ()
    for row in rows:
        for key in ("answer", "prompt"):
            for token in exchange_analysis(row[key])[0].tokens:
                features = dict(token.grammatical_features)
                if (token.lemma in _CONTACT_RESPONSE_ACTIONS
                        and same_lexical_family(token.lemma, verb.lemma)
                        and features.get("person") == "first"
                        and features.get("tense") == "present"):
                    source_id = row[key + "_event_id"]
                    slot = {"binding": "speech_act", "part": "verb", "lemma": token.lemma,
                            "surface": token.surface, "features": features}
                    identifier = hashlib.sha256(token.lemma.encode()).hexdigest()[:24]
                    return (ObservedDialogueForm(
                        (slot,), "fragment", "question", "contact", (source_id,), 1, 0.55,
                        "question", "exchange-part:" + identifier, (), "perform_contact", "core",
                        "exchange-syntax:" + identifier, ("act:0",), None, None, False,
                        "exchange-rule:" + identifier, 0.8, 0.55,
                        tuple(sorted(turn_features(prompt))),
                        ("meaning:perform_contact",), (), (), (), (), (("reaction", 1.0),),
                    ),)
    return ()
