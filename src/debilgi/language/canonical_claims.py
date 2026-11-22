"""Canonical values for PropositionMemory; no separate store or reply policy."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta, tzinfo
from zoneinfo import ZoneInfo

from debilgi.config import thought_content_enabled
from debilgi.events.models import Event
from debilgi.language.productive_nlg import ClauseThought, ConceptArgument
from debilgi.language.semantic_graph import MeaningProof, parse_proposition
from debilgi.language.structured_utterance import SemanticTime, StructuredUtterance

SCHEMA_VERSION = 1
RELATIVE_DAYS = {"позавчера": -2, "вчера": -1, "сегодня": 0, "завтра": 1, "послезавтра": 2}
# Closed-class adverbs of recency and frequency (deictic grammar words, as RELATIVE_DAYS): the agent
# says one only if people of this server are seen to use it (stage 4.33).
RECENCY_ADVERB = "недавно"
FREQUENCY_ADVERBS = frozenset({"часто", "чаще", "обычно", "постоянно"})


def source_kind(event: Event) -> str:
    # Imported dialogue actors are fictional even when they have Discord-shaped IDs.
    if "corpus" in event.source or event.source.startswith("seed:"):
        return "corpus_form"
    return "self_report"


def argument_ref(argument: ConceptArgument) -> str:
    return argument.head.participant_ref or f"concept:{argument.head.lemma}"


def bound_participants(thought: ClauseThought) -> tuple:
    """Instance bindings scope a lexical meaning; they never become synonyms."""
    return tuple(
        sorted(
            (dep.role, slot, word.participant_ref)
            for dep in thought.dependencies
            for slot, words in (
                ("head", (dep.argument.head,)),
                ("modifier", dep.argument.modifiers),
                ("postmodifier", dep.argument.postmodifiers),
            )
            for word in words
            if word.participant_ref
        )
    )


def semantic_time(time: SemanticTime, event: Event, timezone: tzinfo | None) -> SemanticTime:
    local = event.timestamp.astimezone(timezone)
    start = end = None
    if time.deictic in RELATIVE_DAYS:
        start = local.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
            days=RELATIVE_DAYS[time.deictic]
        )
        end = start + timedelta(days=1)
    elif time.deictic == "сейчас":
        start, end = local, local + timedelta(hours=1)
    return replace(
        time,
        anchor=event.utc_timestamp.isoformat(),
        start=start.astimezone(UTC).isoformat() if start else None,
        end=end.astimezone(UTC).isoformat() if end else None,
    )


def fact_thought(structure: StructuredUtterance) -> ClauseThought | None:
    """Reuse the stage-1 adapters, consuming an IR correction only as an update."""
    from debilgi.language.generative_planner import statement_thought

    if structure.corrections:
        # Keep token positions/scopes intact. Only the IR-marked correction
        # operator is consumed; embedded negation remains part of the claim.
        structure = replace(
            structure,
            corrections=(),
            tokens=tuple(
                replace(t, surface="", lemma="") if t.index == 0 and t.lemma == "нет" else t
                for t in structure.tokens
            ),
        )
    thought = statement_thought(structure, speaker="user") or parse_proposition(structure)
    if thought is None:
        return None
    if structure.clauses and structure.clauses[0].subject_ref:
        ref = structure.clauses[0].subject_ref.ref
        thought = replace(
            thought,
            dependencies=tuple(
                replace(
                    d,
                    argument=replace(
                        d.argument, head=replace(d.argument.head, participant_ref=ref)
                    ),
                )
                if d.role == "subject"
                else d
                for d in thought.dependencies
            ),
        )
    return thought


@dataclass(frozen=True, slots=True)
class ClaimQuery:
    guild_id: int
    channel_id: int | None
    speaker_id: int | None
    snapshot_at: str
    exclude_event_id: str
    predicate: str | None = None
    arguments: tuple[tuple[str, str], ...] = ()
    requested_role: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    tense: str | None = None
    deictic: str | None = None
    include_history: bool = False
    include_unresolved: bool = False
    calendar_zone: str | None = None
    predicate_parent: str | None = None
    intent_evidence_ids: tuple[str, ...] = ()
    requested_attribute: bool = False


@dataclass(frozen=True, slots=True)
class CanonicalClaim:
    claim_id: str
    subject_ref: str
    predicate: str
    polarity: int
    source_kind: str
    source_event_id: str
    speaker_id: int
    source_time: str
    valid_from: str | None
    valid_to: str | None
    confidence: float
    thought: ClauseThought
    proof: MeaningProof
    supersedes: tuple[str, ...] = ()
    epistemic_status: str = "fact"


@dataclass(frozen=True, slots=True)
class Known:
    query: ClaimQuery
    claims: tuple[CanonicalClaim, ...]
    status: str = "Known"
    reports: tuple[CanonicalClaim, ...] = ()


@dataclass(frozen=True, slots=True)
class Unknown:
    query: ClaimQuery
    claims: tuple[CanonicalClaim, ...] = ()
    status: str = "Unknown"
    reports: tuple[CanonicalClaim, ...] = ()


def query_from_event(
    event: Event,
    text: str | None = None,
    *,
    timezone: tzinfo | None = None,
    hypotheses=(),
) -> ClaimQuery:
    from debilgi.language.generative_planner import objective_for

    structure = event.meaning
    if text is not None and text != structure.text:
        structure = replace(event, payload={**event.payload, "content": text}).meaning
    objective = objective_for(structure)
    clause = next(
        (c for c in reversed(structure.clauses) if c.clause_kind == "question"),
        next(iter(structure.clauses), None),
    )
    arguments = []
    if clause:
        roles = (
            ("subject", clause.subject_indices),
            ("direct_object", clause.direct_object_indices),
            ("indirect_object", clause.indirect_object_indices),
            ("predicate_complement", clause.predicate_complement_indices),
            ("action_complement", clause.action_complement_indices),
        )
        for role, indices in roles:
            heads = [
                structure.tokens[i]
                for i in indices
                if structure.tokens[i].part_of_speech
                in {"noun", "pronoun", "verb", "participle", "adjective"}
                and structure.tokens[i].semantic_role != "question_target"
            ]
            if len(heads) > 1 and role == "predicate_complement":
                nouns = [t for t in heads if t.part_of_speech == "noun"]
                if len(nouns) == 1:
                    heads = nouns
            if len(heads) == 1:
                token = heads[0]
                arguments.append(
                    (role, token.participant.ref if token.participant else f"concept:{token.lemma}")
                )
        if clause.subject_ref and not any(role == "subject" for role, _ in arguments):
            arguments.append(("subject", clause.subject_ref.ref))
    time = semantic_time(clause.semantic_time, event, timezone) if clause else SemanticTime()
    if objective.predicate is None and clause and clause.implicit_predicate != "быть":
        arguments = [
            ("*", f"concept:{t.lemma}") for t in structure.tokens if t.part_of_speech == "noun"
        ]
    requested = {
        "кто": "subject",
        "что": "direct_object",
        "кому": "indirect_object",
        "где": "location",
        "почему": "cause",
        "зачем": "cause",
    }.get(objective.question_word)
    wh = next((t for t in structure.tokens if t.semantic_role == "question_target"), None)
    attribute = bool(wh and wh.part_of_speech == "adjective")
    if attribute and requested is None:
        requested = "direct_object"
    if wh and wh.index and structure.tokens[wh.index - 1].part_of_speech == "preposition":
        requested = "topic"
    bot_id = event.payload.get("bot_user_id")
    agent_addressed = bool(event.payload.get("bot_mentioned")) or (
        isinstance(bot_id, int)
        and bot_id in {event.target_id, event.payload.get("reply_to_author_id")}
    )
    agent_ref = f"discord:agent:{bot_id}" if isinstance(bot_id, int) else "discord:agent:unresolved"
    addressed_self = (
        agent_addressed
        and (
            any(t.lemma in {"ты", "вы", "твой", "ваш"} for t in structure.tokens)
            or bool(clause and clause.implicit_subject == "addressee")
            or any(t.lemma == "себя" for t in structure.tokens)
        )
        and (
            bool(clause and clause.clause_kind == "question")
            or bool(objective.question_word)
            or any(t.lemma == "себя" for t in structure.tokens)
        )
    )
    predicate = objective.predicate or (
        "быть" if clause and clause.implicit_predicate == "быть" else None
    )
    if clause and any(structure.tokens[i].lemma in {"я", "мы"} for i in clause.subject_indices):
        addressed_self = False
    if any(link.relation == "advcl:condition" for link in structure.clause_links):
        addressed_self = False
    if addressed_self:
        arguments = [(r, v) for r, v in arguments if r != "subject"]
        arguments.append(("subject", agent_ref))
        if any(t.lemma == "звать" for t in structure.tokens):
            predicate, requested = "называться", "predicate_complement"
            arguments = [("subject", agent_ref)]
        if any(t.lemma == "где" for t in structure.tokens):
            requested = "location"
    # These are query operators, not proposed answers. Keep the original
    # predicate as a one-hop parent; never materialize generalized claims.
    parent = None
    # «Чем занимаешься?» asks the same (instrumental of «что»); flag only, stage 4.33.
    # The analyser reads «чем» as a conjunction, so for this verb the word itself is the cue.
    asked_what = objective.question_word == "что" or (
        thought_content_enabled() and predicate == "заниматься"
        and any(t.surface.casefold() == "чем" for t in structure.tokens))
    if predicate in {"делать", "заниматься"} and asked_what:
        parent, predicate, requested = predicate, None, None
    # Nominal location questions license a copula without a lexical predicate.
    if requested == "location" and predicate is None:
        predicate = "быть"
    if objective.question_word == "кто" and addressed_self:
        predicate, requested = "быть", "predicate_complement"
    # Learned intentions can fill an unspecified grammatical slot only when
    # their request predicate and interrogative agree with this turn. Their
    # demonstrated response predicates never become facts or query contents.
    aligned = tuple(
        h
        for h in hypotheses[:8]
        if objective.predicate in h.signature.get("request_predicates", ())
        and objective.question_word == h.signature.get("requested_slot")
    )
    if not requested and not parent and aligned:
        cases = {h.signature.get("requested_case") for h in aligned}
        if len(cases) == 1:
            requested = {"dative": "indirect_object", "accusative": "direct_object"}.get(
                next(iter(cases))
            )
    return ClaimQuery(
        event.guild_id,
        event.channel_id,
        event.actor_id,
        event.utc_timestamp.isoformat(),
        event.event_id,
        predicate,
        tuple(arguments),
        requested,
        time.start,
        time.end,
        time.tense,
        time.deictic,
        calendar_zone=getattr(timezone, "key", None),
        predicate_parent=parent,
        intent_evidence_ids=tuple(dict.fromkeys(i for h in aligned for i in h.evidence_ids))[:8],
        requested_attribute=attribute,
    )


def answer_thought(claim: CanonicalClaim, query: ClaimQuery) -> ClauseThought:
    """Realize the retained interval in the question's temporal perspective."""
    thought = claim.thought
    expression = query.deictic
    if not expression and thought.semantic_time.deictic in RELATIVE_DAYS and claim.valid_from:
        zone = ZoneInfo(query.calendar_zone) if query.calendar_zone else None
        date = datetime.fromisoformat(claim.valid_from).astimezone(zone).date()
        today = datetime.fromisoformat(query.snapshot_at).astimezone(zone).date()
        expression = next(
            (word for word, days in RELATIVE_DAYS.items() if days == (date - today).days),
            date.strftime("%d.%m.%Y"),
        )
    if expression and thought.semantic_time.deictic:
        old = thought.semantic_time.deictic
        thought = replace(
            thought,
            semantic_time=replace(thought.semantic_time, deictic=expression),
            dependencies=tuple(
                replace(
                    d,
                    argument=replace(
                        d.argument,
                        head=replace(d.argument.head, lemma=expression, surface=expression),
                    ),
                )
                if d.role == "manner" and d.argument.head.lemma == old
                else d
                for d in thought.dependencies
            ),
        )
    return replace(
        thought,
        confidence=claim.confidence,
        evidence=tuple(dict.fromkeys((claim.claim_id, *claim.proof.source_event_ids))),
    )


def self_query(query: ClaimQuery) -> bool:
    return any(
        role == "subject" and ref.startswith("discord:agent:") for role, ref in query.arguments
    )


def activity_thought(predicate: str, target: str | None = None) -> ClauseThought:
    """An operation's semantic frame; wording is left to productive morphology.

    An intransitive operation (no ``target``) has the agent as its only argument.
    """
    from debilgi.language.productive_nlg import ConceptLexeme, SemanticDependency

    dependencies = [
        SemanticDependency("subject", ConceptArgument(ConceptLexeme("я", "я", "pronoun")))
    ]
    if target is not None:
        dependencies.append(
            SemanticDependency(
                "direct_object",
                ConceptArgument(
                    ConceptLexeme(target, target, "noun"), grammatical_case="accusative"
                ),
            )
        )
    return ClauseThought(
        ConceptLexeme(predicate, predicate, "verb", (("tense", "past"), ("gender", "masculine"))),
        tuple(dependencies),
        confidence=1.0,
    )
