"""Event-backed working assertions and conservative cross-turn hypotheses.

These are attributed claims, not verified world facts. Rebuilding from retained
events makes the view replayable without a second mutable source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta

from debilgi.events.models import Event, EventType
from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    SemanticDependency,
)
from debilgi.language.structured_utterance import analyze_structured_utterance


@dataclass(frozen=True, slots=True)
class Assertion:
    source_event_id: str
    speaker: str
    thought: ClauseThought
    proposition_key: tuple
    temporarily_bounded: bool
    # Confidence describes parsing, never the truth of a speaker's claim.
    parsing_confidence: float


@dataclass(frozen=True, slots=True)
class AssertionLink:
    kind: str
    earlier_event_id: str
    later_event_id: str


@dataclass(frozen=True, slots=True)
class AssertionMemory:
    assertions: tuple[Assertion, ...] = ()
    links: tuple[AssertionLink, ...] = ()

    @property
    def active(self) -> tuple[Assertion, ...]:
        latest = {}
        for assertion in self.assertions:
            latest[(assertion.speaker, assertion.proposition_key)] = assertion
        return tuple(latest.values())


@dataclass(frozen=True, slots=True)
class ContinuationPlan:
    kind: str
    thought: ClauseThought
    source_event_ids: tuple[str, ...]
    confidence: float


_MESSAGES = {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED, EventType.BOT_REPLIED}
_UNSUPPORTED = {"если", "бы", "либо", "или", "возможно", "наверное", "кажется"}
_TEMPORARY = {"пока", "ещё", "еще"}
_EVENT_TIMES = {"вчера", "сегодня", "позавчера", "недавно", "давно", "утром", "вечером", "ночью"}


def _own_reply(item: Event, current: Event) -> bool:
    return item.event_type == EventType.BOT_REPLIED or (
        item.payload.get("author_is_bot") is True
        and current.payload.get("bot_user_id") is not None
        and item.actor_id == current.payload.get("bot_user_id")
    )


def _scoped_events(events: tuple[Event, ...], current: Event) -> tuple[Event, ...]:
    if current.actor_id is None or current.channel_id is None:
        return ()
    unique = {
        item.event_id: item
        for item in events
        if item.event_id != current.event_id
        and item.guild_id == current.guild_id
        and item.channel_id == current.channel_id
        and item.event_type in _MESSAGES
        and current.utc_timestamp - timedelta(minutes=5)
        <= item.utc_timestamp
        < current.utc_timestamp
        and (
            item.actor_id == current.actor_id
            and not item.payload.get("author_is_bot")
            or _own_reply(item, current)
            and item.target_id == current.actor_id
        )
    }
    return tuple(sorted(unique.values(), key=lambda item: (item.utc_timestamp, item.event_id)))[
        -12:
    ]


def _claim(event: Event, current: Event) -> Assertion | None:
    text = event.payload.get("content")
    if not isinstance(text, str):
        return None
    bot_id = current.payload.get("bot_user_id")
    if isinstance(bot_id, int):
        text = text.replace(f"<@{bot_id}>", " ").replace(f"<@!{bot_id}>", " ")
    if any(mark in text for mark in ('"', "«", "»", "`", ">")):
        return None
    structure = analyze_structured_utterance(text)
    if len(structure.clauses) != 1:
        return None
    clause = structure.clauses[0]
    words = {token.lemma for token in structure.tokens}
    if (
        clause.clause_kind != "statement"
        or clause.predicate_index is None
        or len(clause.subject_indices) != 1
        or clause.coordinations
        or len(clause.complements) > 1
        or any(len(c.object_indices) != 1 or not (
            c.role == "location" or c.role == "location_or_destination"
            and dict(structure.tokens[c.object_indices[0]].grammatical_features).get("case")
            in {"prepositional", "accusative"}
        ) for c in clause.complements)
        or words & _UNSUPPORTED
        or any(t.lemma in {"что", "чтобы", "потому", "когда"} for t in structure.tokens)
    ):
        return None
    bot = _own_reply(event, current)
    speaker = "agent" if bot else f"user:{event.actor_id}"
    # A user's unaddressed 'ты' may refer to another human.
    addressed = (
        bot or event.target_id == current.payload.get("bot_user_id") and event.target_id is not None
    )
    addressed = addressed or bool(event.payload.get("bot_mentioned"))

    def lexeme(index: int) -> ConceptLexeme | None:
        token = structure.tokens[index]
        lemma, surface = token.lemma, token.surface
        features = dict(token.grammatical_features)
        if token.morphology_known is False:
            features["morphology_guessed"] = "true"
        entity = None
        if lemma in {"я", "ты"}:
            if lemma == "ты" and not addressed:
                return None
            entity = speaker if lemma == "я" else (f"user:{current.actor_id}" if bot else "agent")
            lemma = "я" if entity == "agent" else "ты"
            surface = lemma
            features["person"] = "first" if entity == "agent" else "second"
        elif lemma in {"он", "она", "оно", "они", "мы", "вы", "это"}:
            return None  # unresolved referents must not merge unrelated claims
        return ConceptLexeme(
            lemma, surface, token.part_of_speech, tuple(sorted(features.items())), entity
        )

    role_indices = {
        "subject": clause.subject_indices,
        "direct_object": clause.direct_object_indices,
        "indirect_object": clause.indirect_object_indices,
        "action_complement": clause.action_complement_indices,
        "predicate_complement": clause.predicate_complement_indices,
    }
    used = {clause.predicate_index}
    dependencies = []
    for role, indices in role_indices.items():
        if not indices:
            continue
        if len(indices) != 1:
            return None
        head = lexeme(indices[0])
        if head is None:
            return None
        used.update(indices)
        dependencies.append(SemanticDependency(role, ConceptArgument(head)))
    for complement in clause.complements:
        index = complement.object_indices[0]
        head = lexeme(index)
        if head is None:
            return None
        used.update((complement.preposition_index, index))
        dependencies.append(SemanticDependency("location", ConceptArgument(
            replace(head, features=(*head.features, ("indeclinable", "true"),
                ("location_kind", "destination" if complement.role == "location_or_destination"
                 and dict(head.features).get("case") == "accusative" else "location"))),
            preposition=structure.tokens[complement.preposition_index].lemma,
            grammatical_case=dict(head.features).get("case"),
        )))
    # The parser leaves instrumental predicative adjectives unassigned.
    # Preserve their meaning instead of silently dropping a self-assessment.
    for token in structure.tokens:
        if token.part_of_speech is None and token.surface[:1].isalnum():
            return None
        if token.index in used or token.part_of_speech not in {
            "noun",
            "verb",
            "adjective",
            "adverb",
            "pronoun",
            "number",
        }:
            continue
        if token.lemma in _TEMPORARY:
            continue
        if token.surface in _EVENT_TIMES or token.lemma in _EVENT_TIMES:
            if any(d.role == "manner" for d in dependencies):
                return None
            dependencies.append(SemanticDependency("manner", ConceptArgument(
                ConceptLexeme(token.lemma, token.surface, "adverb", (("indeclinable", "true"),)),
            )))
            used.add(token.index)
            continue
        if (
            token.part_of_speech == "adjective"
            and dict(token.grammatical_features).get("case") == "instrumental"
            and structure.tokens[clause.predicate_index].lemma == "считать"
            and not clause.predicate_complement_indices
        ):
            head = lexeme(token.index)
            dependencies.append(
                SemanticDependency(
                    "predicate_complement", ConceptArgument(head, grammatical_case="instrumental")
                )
            )
            used.add(token.index)
        else:
            return None  # never turn a qualified claim into a broader assertion
    predicate = lexeme(clause.predicate_index)
    if predicate is None:
        return None
    features = dict(predicate.features)
    subject = dependencies[0].argument.head
    if subject.entity_kind and features.get("tense") != "past":
        features["person"] = dict(subject.features)["person"]
        predicate = replace(predicate, features=tuple(sorted(features.items())))
    thought = ClauseThought(
        predicate=predicate,
        dependencies=tuple(dependencies),
        negated=clause.negated,
        confidence=clause.confidence,
        evidence=(f"assertion_source:{event.event_id}", f"speaker:{speaker}"),
        predicate_class="modal" if clause.action_complement_indices else "lexical",
    )
    key = (
        predicate.lemma,
        tuple(
            (name, value)
            for name, value in predicate.features
            if name in {"tense", "mood", "aspect"}
        ),
        tuple(
            (
                dep.role,
                dep.argument.head.entity_kind or dep.argument.head.lemma,
                dict(dep.argument.head.features).get("number"),
            )
            for dep in dependencies
        ),
    )
    return Assertion(
        event.event_id, speaker, thought, key, bool(words & _TEMPORARY), clause.confidence
    )


def build_assertion_memory(events: tuple[Event, ...], current: Event) -> AssertionMemory:
    assertions = []
    links = []
    for item in _scoped_events(events, current):
        claim = _claim(item, current)
        if claim is None:
            continue
        compared_speakers = set()
        for earlier in reversed(assertions):
            if (
                earlier.proposition_key != claim.proposition_key
                or earlier.speaker in compared_speakers
            ):
                continue
            compared_speakers.add(earlier.speaker)
            same_polarity = earlier.thought.negated == claim.thought.negated
            kind = (
                ("repeated_claim" if same_polarity else "revised_position")
                if earlier.speaker == claim.speaker
                else ("agreement" if same_polarity else "disagreement")
            )
            links.append(AssertionLink(kind, earlier.source_event_id, claim.source_event_id))
        assertions.append(claim)
    return AssertionMemory(tuple(assertions), tuple(links))


def plan_assertion_continuation(
    prompt: str,
    memory: AssertionMemory,
    events: tuple[Event, ...],
    current: Event,
) -> ContinuationPlan | None:
    """Test a possible temporal change; never assert it as an observed fact."""
    structure = analyze_structured_utterance(prompt)
    words = {t.lemma for t in structure.tokens if t.surface[:1].isalnum()}
    temporal_ellipsis = bool(words & {"теперь", "сейчас"}) and words <= {
        "а",
        "ну",
        "и",
        "теперь",
        "сейчас",
    }
    temporal_arrival = (
        bool(words & {"время", "момент"})
        and bool(words & {"настать", "наступить"})
        and words <= {"ну", "всё", "все", "вот", "и", "время", "момент", "настать", "наступить"}
        and not any(c.negated for c in structure.clauses)
    )
    if not (temporal_ellipsis or temporal_arrival):
        return None
    recent = _scoped_events(events, current)[-6:]
    recent_ids = {item.event_id for item in recent}
    candidates = [
        claim
        for claim in memory.active
        if claim.temporarily_bounded and claim.source_event_id in recent_ids
    ]
    if len(candidates) != 1:
        return None
    chosen = candidates[0]
    # Do not revive an old topic across an unrelated intervening assertion.
    if any(
        claim.proposition_key != chosen.proposition_key
        for claim in memory.active
        if claim.source_event_id in recent_ids
    ):
        return None
    confidence = min(0.82, chosen.parsing_confidence)
    evidence = (
        *chosen.thought.evidence,
        f"continuation_source:{current.event_id}",
        "hypothesis:temporal_change_not_confirmed",
    )
    thought = replace(
        chosen.thought,
        clause_kind="question",
        negated=not chosen.thought.negated,
        response_polarity=None,
        confidence=confidence,
        evidence=evidence,
    )
    return ContinuationPlan(
        "check_temporal_change", thought, (chosen.source_event_id, current.event_id), confidence
    )


def recall_event_slot(prompt: str, memory: AssertionMemory, current: Event) -> ClauseThought | None:
    """Answer a literal time/place question from an attributed recent event."""
    structure = analyze_structured_utterance(prompt)
    if len(structure.clauses) != 1:
        return None
    frame = structure.clauses[0]
    if (frame.question_target_index is None or frame.predicate_index is None
            or len(frame.subject_indices) != 1 or frame.modal_lemmas
            or frame.coordinations or frame.negated):
        return None
    target = structure.tokens[frame.question_target_index].lemma
    wanted_role = {"когда": "manner", "где": "location", "куда": "location"}.get(target)
    if wanted_role is None:
        return None
    subject = structure.tokens[frame.subject_indices[0]].lemma
    if subject != "я":
        return None
    predicate = structure.tokens[frame.predicate_index].lemma
    if dict(structure.tokens[frame.predicate_index].grammatical_features).get("tense") != "past":
        return None
    objects = [structure.tokens[i] for i in frame.direct_object_indices]
    if len(objects) != 1:
        return None
    obj = objects[0]
    used = {frame.question_target_index, frame.predicate_index,
            *frame.subject_indices, *frame.direct_object_indices}
    if any(t.index not in used and t.part_of_speech in {"noun", "adjective", "adverb", "verb"}
           for t in structure.tokens):
        return None
    matches = []
    for claim in reversed(memory.assertions):
        if (claim.speaker != f"user:{current.actor_id}"
                or claim.thought.predicate.lemma != predicate):
            continue
        reply_to = current.payload.get("reply_to_message_id")
        if reply_to is not None and claim.source_event_id != f"discord:message:{reply_to}":
            continue
        roles = {d.role: d.argument for d in claim.thought.dependencies}
        if (roles.get("subject") is None
                or roles["subject"].head.entity_kind != f"user:{current.actor_id}"
                or "direct_object" not in roles):
            continue
        antecedent = roles["direct_object"].head
        if obj.part_of_speech == "pronoun":
            if obj.lemma not in {"он", "она", "оно", "они", "это"}:
                continue
            requested_number = dict(obj.grammatical_features).get("number")
            if dict(antecedent.features).get("morphology_guessed") != "true":
                if requested_number and requested_number != dict(antecedent.features).get("number"):
                    continue
                requested_gender = dict(obj.grammatical_features).get("gender")
                if requested_gender and requested_gender != dict(antecedent.features).get("gender"):
                    continue
        elif obj.lemma != antecedent.lemma:
            continue
        matches.append((claim, roles))
    # Ambiguous pronouns must not silently select one of several objects.
    if not matches or (obj.part_of_speech == "pronoun" and len({
        roles["direct_object"].head.lemma for _, roles in matches
    }) != 1):
        return None
    claim, roles = matches[0]
    if claim.thought.negated or wanted_role not in roles:
        return None
    if wanted_role == "location":
        destination = dict(roles["location"].head.features).get("location_kind") == "destination"
        if (target == "куда") != destination:
            return None
    return replace(claim.thought, confidence=0.92, evidence=(
        *claim.thought.evidence, f"event_slot:{target}", "source_kind:user_report",
    ))


def event_followup_thought(event: Event) -> ClauseThought | None:
    """Ask about a missing event slot while retaining an opaque object name."""
    claim = _claim(event, event)
    if claim is None or claim.thought.negated or claim.thought.predicate_class != "lexical":
        return None
    roles = {d.role: d.argument for d in claim.thought.dependencies}
    if ("direct_object" not in roles
            or dict(claim.thought.predicate.features).get("tense") != "past"):
        return None
    if "manner" not in roles:
        role, question = "manner", "когда"
    elif "location" not in roles:
        role, question = "location", "где"
    else:
        return None
    return replace(claim.thought, clause_kind="question", focus_role=role,
                   dependencies=(*claim.thought.dependencies, SemanticDependency(
                       role, ConceptArgument(ConceptLexeme(question, question, "adverb")),
                   )), evidence=(*claim.thought.evidence, "partial_event:missing_slot"))
