"""Learn choices between meanings from consequences of actually sent actions.

No response strings, desired topics, or intrinsic positive action values live
here. Cold starts explore equally; nonzero values come only from attributed
human feedback. The preference is selected and stored before surface wording.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta

from debilgi.events.models import Event, EventType
from debilgi.experience.feedback import DEFAULT_FEEDBACK_ATTRIBUTOR
from debilgi.language.dialogue_obligations import resolve_dialogue_exchange, thought_from_data
from debilgi.language.generative_planner import DerivedPlan, DialogueObjective
from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    SemanticDependency,
    identity_graph,
)
from debilgi.language.semantic_graph import MeaningGoal, proposition_key
from debilgi.language.tokenizer import tokenize
from debilgi.social.signals import AFFIRMATION, REFUSAL
from debilgi.storage.database import Database


@dataclass(frozen=True, slots=True)
class MeaningSelection:
    intention_id: str
    proposal: DerivedPlan
    expected_reward: float
    expected_outcome: str | None
    sample_count: int
    alternatives: tuple[dict, ...]
    exploring: bool
    choice: dict = field(default_factory=dict)


_BASIS_KEYS = ("stance_evidence", "stance_consistency", "human_answer_people", "human_answer_match",
               "affect_events", "affect_basis", "revises", "revised_because", "fantasy", "boredom",
               "irritation", "curiosity_target", "opinion")


def delivered_basis(premises=(), metadata=None) -> dict:
    """What the agent actually had when it chose this turn (its own decision record).

    Premises are the semantic evidence of an ordinary plan; a night hypothesis adds the
    stance evidence the day arbiter computed. It is stored next to the delivered turn and
    never becomes a fact: only a reader asked "why?" may restate it.
    """
    metadata = metadata or {}
    basis = {"premises": [{"evidence_id": p.evidence_id, "kind": p.kind, "text": p.text,
                           "thought": identity_graph(p.thought)} for p in premises or ()]}
    basis.update({k: metadata[k] for k in _BASIS_KEYS if metadata.get(k) not in (None, [], ())})
    if metadata.get("derivation"):
        basis["derivation"] = metadata["derivation"]
    return basis


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


# ----------------------------------------------------------------------------- boredom (4.33)
# Boredom B is a function of the journal of the agent's own delivered turns and the clock, never a
# stored number: B = 1 - exp(-sensitivity * sum_i w_i * s_i), w_i = 2^(-age_i / half_life). The
# constants come from the genome: sensitivity = novelty_seeking, half_life = (1 - patience) of a
# day (an impatient character forgets a repeat sooner). The day is the one hand-set unit: it is
# the grain of the calendar words the agent can say («сегодня», «вчера»). Under
# ``DEBILGI_THOUGHT_CONTENT`` only.
BOREDOM_DAY = 86400.0
# How many decayed repeats of one meaning make the agent tired of it: the third time in a row
# (two fresh repeats weigh a little under 2). A hand-set starting point; nothing learns it yet.
TIRED_REPEATS = 1.5
BOREDOM_PENALTY = 3.0
# What was said to another person of the same channel counts half: the room heard it, but this
# person did not (hand-set).
PEER_WEIGHT = 0.5  # a single fresh repeat costs about the 1.5 the repeat counter cost before
_TERMS_CACHE: dict[str, tuple] = {}


# A demand the person presses again (a question to the agent's reaction, a repeat, «почему»,
# «какое», «понял?») was not closed by that reaction (4.41, flag only). Experience of it is the
# frequency of this outcome for this operation, in this context, for this person; no reward.
UNCLOSED_OUTCOMES = frozenset({"follow_up_question"})


def social_half_life() -> float:
    """T_S of plan 4.36 §2.2: a day over (1 - memory_attachment) of the genome (≈ 3.6 days) — the
    horizon over which what a person did to the agent is still held against or for them."""
    from debilgi.personality.genome import DEBILGI_GENOME

    return BOREDOM_DAY / max(1e-3, 1.0 - DEBILGI_GENOME.traits()["memory_attachment"])


def _age(row, now) -> float:
    return max(0.0, (now - datetime.fromisoformat(row["observed_at"])).total_seconds())


def unclosed_share(effects, now: datetime | None = None, half_life: float | None = None
                   ) -> tuple[float, tuple[str, ...]]:
    """Share of earlier outcomes that left the demand open, as k/(n+1) (a count that never
    reaches 1: one pressing is half a reason, a habit approaches one), and their intention ids.

    With a clock (4.42) every outcome weighs 2^(-age/T_S) (T_S, the horizon of the person, not of
    the minute): a pressing of last week is a smaller reason than one of this hour. Without it,
    the plain count of 4.41."""
    def weight(row) -> float:
        return 1.0 if now is None else decay_weight(_age(row, now), half_life or social_half_life())

    pressed = [row for row in effects if row["outcome_type"] in UNCLOSED_OUTCOMES]
    if not pressed:
        return 0.0, ()
    return (sum(weight(r) for r in pressed) / (sum(weight(r) for r in effects) + 1.0),
            tuple(dict.fromkeys(row["intention_id"] for row in pressed)))


@dataclass(frozen=True, slots=True)
class Sympathy:
    """S of plan 4.36 §2.2 (4.45, flag only): the agent's own attitude to a topic, a function of
    its own journals and the clock. Never a fact; never written anywhere (the stance that the
    agent takes on its strength is written only when it is delivered)."""

    value: float = 0.0
    people: int = 0
    outcomes: int = 0
    deeds: int = 0
    ids: tuple[str, ...] = ()
    weights: tuple[float, ...] = ()
    newest: str | None = None
    deed_claim: str | None = None  # the latest own deed that stands for the topic (its claim id)

    def as_data(self) -> dict:
        return {"value": round(self.value, 4), "people": self.people, "outcomes": self.outcomes,
                "deeds": self.deeds, "ids": list(self.ids[:8]),
                "warmth_weights": [round(w, 3) for w in self.weights]}


def _same_topic(wanted: frozenset, term: str) -> bool:
    """The term is the topic (or its aspect partner: «изучать» / «изучить»)."""
    if term in wanted:
        return True
    from debilgi.language.dialogue_learning import _same_root_pair
    from debilgi.language.lexical_matching import aspect_pair

    return any(w[:3] == term[:3] and aspect_pair(w, term)
               and (_same_root_pair(w, term) or _same_root_pair(term, w)) for w in wanted)


def decay_weight(age_seconds: float, half_life: float) -> float:
    """2^(-age/T): the only way time enters a scale (no window, no draw)."""
    return 2.0 ** (-max(0.0, age_seconds) / half_life)


# ---- the person (4.42, flag only): hypotheses about how THIS person receives a kind of move ---
# A hypothesis is a prediction, never a label: «after a move of kind m this person reacts r». It
# lives in continuous_cognition_hypotheses (subject = the person, status 'hypothesis'), is
# checked by the person's next reactions (continuous_cognition_evidence) and is never a claim.
PERSON_SIGNATURE = "person:"
# an answer of a person to the agent's question (4.44): his position, keyed by the topic
REPORT_KEY = "report:"
REPORT_SCOPE = "report"
# the kind of move is a reading of the agent's own operation (its plan), not of words
_COUNTER_ACTS = frozenset({"slot_question", "reciprocal_question", "check_condition"})
_COUNTER_OPERATIONS = frozenset({"curiosity", "repair", "ask_unbound_variable"})
_PLAY_ACTS = frozenset({"mirror_amusement", "retort"})
# reactions of the person to a move: «pressed» left the demand open (weighed already as `unclosed`),
# «laughed», «hostile» (hostility cues), «rejected» (explicit rejection / correction), «continued»
RECEPTIONS = ("pressed", "laughed", "hostile", "rejected", "continued", "answered", "evaded")
# a move taken at face value («serious»: an answer, or the honest limit) is not meant to be laughed
# at; no move is meant to be met with hostility or rejection
_UNWELCOME = frozenset({"hostile", "rejected", "evaded"})
# what the reaction says about the person's attitude to the agent's action: warm, neutral, cold
_WARMTH = {"continued": 1.0, "laughed": 0.5, "pressed": 0.0, "rejected": 0.0, "hostile": 0.0,
           "answered": 1.0, "evaded": 0.5}
WARMTH_RATE = 0.15  # the rate cognition_entity_models already uses for reply_probability


def move_type(operation: str, hypothesis_kind: str | None = None) -> str:
    """serious (an answer or the honest limit) / reaction / counter (a question back) / play."""
    if operation.startswith("compose_reaction:"):
        act = operation.split(":", 1)[1]
        return ("play" if act in _PLAY_ACTS else "counter" if act in _COUNTER_ACTS
                else "reaction")
    if hypothesis_kind == "fantasy":
        return "play"
    if operation == "acknowledge_report":
        return "reaction"
    if operation == "opinion":
        return "stance"  # (4.45) a view of the agent's own, to be met or tested by the person
    if operation in _COUNTER_OPERATIONS or operation.startswith("clarify_"):
        return "counter"
    return "serious"


def reaction_class(text: str, outcome: str, hostile: bool) -> str:
    from debilgi.language.dialogue_learning import reactive_act

    if outcome in UNCLOSED_OUTCOMES:
        return "pressed"
    if hostile:
        return "hostile"
    if outcome in {"explicit_rejection", "contextual_correction", "refusal", "negative_reaction"}:
        return "rejected"
    if text and reactive_act(text) == "amusement":
        return "laughed"
    if outcome in {"answered", "requested_information_received"}:
        return "answered"
    return "evaded" if outcome == "evaded" else "continued"


def reception_cost(move: str, trust: dict[str, float]) -> float:
    """What the forecast says this kind of move costs with this person: the weight of reactions
    the move is never meant to get (hostility, rejection; for a serious move also laughter).
    «pressed» is not counted here: `unclosed` already weighs it."""
    unwelcome = _UNWELCOME | ({"laughed"} if move in {"serious", "stance"} else set())
    return sum(trust.get(reaction, 0.0) for reaction in unwelcome)


async def reweigh_person(database: Database, hypothesis_ids, now: datetime) -> int:
    """Trust of person hypotheses from their evidence, each weighing 2^(-age/T_S): the day-time
    update and the night's recount are the same function. Evidence counts are the raw counts."""
    half_life = social_half_life()
    changed = 0
    for hypothesis_id in hypothesis_ids:
        rows = await database.fetchall(
            "SELECT relation, observed_at FROM continuous_cognition_evidence WHERE hypothesis_id=?",
            (hypothesis_id,))
        weights = {"supports": 0.0, "contradicts": 0.0}
        counts = {"supports": 0, "contradicts": 0}
        for row in rows:
            counts[row["relation"]] += 1
            weights[row["relation"]] += decay_weight(_age(row, now), half_life)
        confidence = weights["supports"] / (weights["supports"] + weights["contradicts"] + 1.0)
        await database.execute(
            "UPDATE continuous_cognition_hypotheses SET confidence=?, positive_evidence=?,"
            " negative_evidence=?, last_reviewed_at=? WHERE hypothesis_id=?",
            (min(1.0, max(0.0, confidence)), counts["supports"], counts["contradicts"],
             now.isoformat(), hypothesis_id))
        changed += 1
    return changed


@dataclass(frozen=True, slots=True)
class BoredomRow:
    intention_id: str
    age: float
    behavior_key: str
    operation: str
    terms: frozenset
    text_key: str
    receipt_id: str
    text: str
    scope: str | None = None
    person: bool = True  # said to the one who asks now (else to someone else in the channel)
    material: frozenset = frozenset()  # what the reaction was about: goal topics and roles


@dataclass(frozen=True, slots=True)
class BoredomReading:
    value: float
    meaning: float
    surface: float
    repeats: float  # weighted number of earlier equal meanings (the sum inside the exponent)
    sources: tuple[str, ...]
    same_plan: float = 0.0  # the part of it that is the very same plan (equal behavior key)

    def as_data(self) -> dict:
        return {"value": round(self.value, 4), "meaning": round(self.meaning, 4),
                "surface": round(self.surface, 4), "repeats": round(self.repeats, 4),
                "same_plan": round(self.same_plan, 4), "sources": list(self.sources)}


class BoredomLedger:
    def __init__(self, rows: tuple[BoredomRow, ...], sensitivity: float, half_life: float) -> None:
        self.rows = rows
        self.sensitivity = sensitivity
        self.half_life = half_life

    def reading(self, *, behavior_key: str | None = None, operation: str | None = None,
                terms=(), text: str | None = None, scope: str | None = None,
                material=frozenset()) -> BoredomReading:
        from debilgi.language.dialogue_learning import normalized_response

        mine = frozenset(terms)
        wanted = normalized_response(text) if text else None
        same, surface, sources, identical = 0.0, 0.0, [], 0.0
        for row in self.rows:
            weight = decay_weight(row.age, self.half_life) * (1.0 if row.person else PEER_WEIGHT)
            similar = 0.0
            if behavior_key is not None and row.behavior_key == behavior_key:
                # The same skeleton about other material (a question about a magazine after
                # one about a book) is another turn, not a repeat of it.
                similar = (len(material & row.material) / len(material | row.material)
                           if material and row.material else 1.0)
                identical += weight * similar
            elif scope is not None and row.scope == scope and operation == row.operation:
                # Another wording of the same not-knowing of the same question is the same thing.
                similar = 1.0
            elif scope is None and operation == row.operation and mine and row.terms:
                # The same material under another meaning: «обработал сообщение» after
                # «отправил сообщение» is a smaller repeat than the same thing said again.
                similar = 0.5 * len(mine & row.terms) / len(mine | row.terms)
            said = wanted is not None and row.text_key == wanted
            if said:
                surface += weight
            if similar or said:
                same += weight * similar
                sources.append((weight * max(similar, float(said)), row.intention_id))
        pull = self.sensitivity
        meaning = 1.0 - math.exp(-pull * same)
        words = 1.0 - math.exp(-pull * surface)
        return BoredomReading(
            max(meaning, words), meaning, words, same,
            tuple(i for _, i in sorted(sources, reverse=True)[:8]), identical)

    def latest(self, behavior_key: str) -> BoredomRow | None:
        return next((r for r in self.rows if r.behavior_key == behavior_key), None)


def limit_scope(query) -> str:
    """What a knowledge limit is about: the asked predicate and arguments, no clocks or ids."""
    data = query if isinstance(query, dict) else asdict(query)
    skip = {"snapshot_at", "exclude_event_id", "intent_evidence_ids", "channel_id", "guild_id"}
    return _json({k: v for k, v in data.items() if k not in skip})


def claim_terms(claim) -> frozenset:
    """Content lemmas of a thought: its predicate and the heads of its arguments."""
    from debilgi.language.content import content_lemmas

    return frozenset(content_lemmas(" ".join(
        (claim.predicate.lemma, *(d.argument.head.lemma for d in claim.dependencies)))))


def _plan_terms(plan_json: str) -> frozenset:
    from debilgi.language.content import content_lemmas

    claim = (json.loads(plan_json).get("claim") or {})
    words = [(claim.get("predicate") or {}).get("lemma") or ""]
    words += [((d.get("argument") or {}).get("head") or {}).get("lemma") or ""
              for d in claim.get("dependencies") or ()]
    return frozenset(content_lemmas(" ".join(w for w in words if w)))


def goal_material(goal) -> frozenset:
    """Topics and bound roles of a reader goal (object or stored dict), as content lemmas."""
    from debilgi.language.content import content_lemmas

    data = goal if isinstance(goal, dict) else asdict(goal)
    words = [*(data.get("topics") or ()),
             *(r[-1] for r in data.get("known_roles") or () if r and isinstance(r[-1], str))]
    return frozenset(content_lemmas(" ".join(w for w in words if isinstance(w, str))))


def _plan_scope(plan_json: str) -> str | None:
    limit = json.loads(plan_json).get("knowledge_limit")
    return limit_scope(limit["query"]) if limit and limit.get("query") else None


def behavior_key(proposal: DerivedPlan) -> str:
    """Semantic identity excludes inflections, source labels and confidence."""
    plan = proposal.plan
    atoms = tuple(
        proposition_key(thought)
        for thought in (
            plan.claim,
            plan.reason,
            plan.condition,
            plan.follow_up,
            plan.reason_condition,
            *plan.additional_claims,
        )
        if thought is not None
    )
    # Epistemic limits and social acts have semantic arguments too. An unknown
    # location must not share a behavior identity with unknown reading history.
    scope = None
    if plan.knowledge_limit:
        query = asdict(plan.knowledge_limit.query)
        scope = {k: v for k, v in query.items()
                 if k not in {"snapshot_at", "exclude_event_id", "intent_evidence_ids"}}
    identity = (proposal.operation, atoms)
    if proposal.operation == "curiosity" and proposal.premises:
        # An inquiry is the same behaviour whatever its wording (alone, or after a limit): its
        # identity is the topic asked about.
        return hashlib.sha256(_json((proposal.operation, proposal.premises[0].text)).encode()
                              ).hexdigest()[:24]
    if scope is not None or plan.speech_act or plan.clarification:
        identity = (*identity, scope, plan.speech_act,
                    asdict(plan.clarification) if plan.clarification else None)
    return hashlib.sha256(_json(identity).encode()).hexdigest()[:24]


def response_behavior(decision):
    """Compact receipt semantics; traces remain diagnostics rather than state."""
    if decision is None:
        return {}
    data = asdict(decision) if not isinstance(decision, dict) else decision
    return {k: v for k, v in data.items() if k != "selection_context"}


class MeaningIntentRepository:
    def __init__(
        self,
        database: Database,
        *,
        exploration_rate: float = 0.10,
        feedback_window: timedelta = timedelta(minutes=10),
        cognition=None,
    ) -> None:
        if not 0 <= exploration_rate <= 1:
            raise ValueError("exploration_rate must be between zero and one")
        self._database = database
        self._cognition = cognition  # the entity models (P: warmth of a person to the agent)
        self._exploration_rate = exploration_rate
        self._feedback_window = feedback_window

    async def irritation(self, event: Event, text: str):
        """R of this person towards the agent (flag only); see cognition.affect."""
        from debilgi.cognition.affect import irritation_reading

        return await irritation_reading(self._database, event, text)

    async def boredom_ledger(self, event: Event) -> BoredomLedger | None:
        """The agent's own delivered turns in this channel, with ages and addressees (flag only)."""
        from debilgi.config import thought_content_enabled
        from debilgi.language.dialogue_learning import normalized_response
        from debilgi.personality.genome import DEBILGI_GENOME

        if not thought_content_enabled() or event.actor_id is None or event.channel_id is None:
            return None
        traits = DEBILGI_GENOME.traits()
        half_life = max(1e-3, 1.0 - traits["patience"]) * BOREDOM_DAY
        now = event.utc_timestamp
        rows = await self._database.fetchall(
            """SELECT intention_id, behavior_key, operation, response_text, plan_json, sent_at,
                      channel_id, response_message_id, user_id
               FROM language_meaning_intentions WHERE guild_id=? AND channel_id=?
               AND response_message_id IS NOT NULL AND sent_at < ? AND sent_at >= ?
               ORDER BY sent_at DESC LIMIT 64""",
            (event.guild_id, event.channel_id, now.isoformat(),
             (now - timedelta(seconds=6 * half_life)).isoformat()))
        entries = []
        for row in rows:
            cached = _TERMS_CACHE.get(row["intention_id"])
            if cached is None:
                cached = (_plan_terms(row["plan_json"]), _plan_scope(row["plan_json"]),
                          goal_material(json.loads(row["plan_json"]).get("goal") or {}))
                if len(_TERMS_CACHE) > 4096:
                    _TERMS_CACHE.clear()
                _TERMS_CACHE[row["intention_id"]] = cached
            terms, scope, material = cached
            text = row["response_text"] or ""
            entries.append(BoredomRow(
                row["intention_id"], (now - datetime.fromisoformat(row["sent_at"])).total_seconds(),
                row["behavior_key"], row["operation"], terms, normalized_response(text),
                f"receipt:{row['channel_id']}:{row['response_message_id']}", text, scope,
                row["user_id"] == event.actor_id, material))
        return BoredomLedger(tuple(entries), traits["novelty_seeking"], half_life)

    async def choose(
        self,
        event: Event,
        proposals: tuple[DerivedPlan, ...],
        objective: DialogueObjective,
        *,
        parent_intent_id: str,
        state_features: tuple[tuple[str, float], ...] = (),
        request_text: str | None = None,
        query_result=None,
        response_decision=None,
    ) -> MeaningSelection | None:
        from debilgi.language.communication import plan_issues
        from debilgi.language.dialogue_learning import reactive_act
        from debilgi.language.generative_planner import _supports
        from debilgi.language.semantic_graph import matches_proposition, meaning_query

        request = meaning_query(request_text or event.meaning.text)

        def canonical_answer(proposal):
            records = tuple(p for p in proposal.premises if p.kind == "canonical_claim")
            if not records:
                return False
            allowed = {c.claim_id for c in query_result.claims} if query_result else set()
            return all(
                p.evidence_id in allowed and p.thought == proposal.plan.claim for p in records
            )

        def conditional_answer(proposal):
            condition = proposal.plan.condition
            return bool(
                condition and request.assumption and proposal.premises
                and any(p.kind.startswith("learned_") for p in proposal.premises)
                and matches_proposition(request.assumption, condition)
            )

        def causal_answer(proposal):
            from dataclasses import replace

            return bool(
                proposal.plan.reason and proposal.premises
                and objective.question_word in {"почему", "зачем"}
                and _supports(replace(objective, question_word=None), proposal.plan.claim)
            )

        def committed_reaction(proposal):
            return bool(
                response_decision is not None and response_decision.mode == "free_form"
                and proposal.plan.speech_act == response_decision.act
                and proposal.plan.claim.predicate_class == "dialogue_act"
                and not proposal.plan.claim.dependencies
                and proposal.plan.claim.evidence
                and proposal.operation.startswith("compose_reaction:")
            )

        def delivered_support(proposal):
            return any(p.kind == "sent_turn" and p.evidence_id.startswith("receipt:")
                       and p.evidence_id in proposal.plan.claim.evidence for p in proposal.premises)

        # A game is offered next to the honest limit, which stays admissible whatever request
        # the focus kept (the limit is what the agent says if the game is not taken).
        playing = any(dict(p.plan.hypothesis).get("kind") == "fantasy" for p in proposals)
        # a view of its own (4.45) is offered next to the honest limit as a game is
        opining = any(p.operation == "opinion" for p in proposals)

        # Reward ranks admissible actions; it cannot grant relevance or truth.
        proposals = tuple(p for p in proposals if not plan_issues(p.plan) and (
            objective.kind not in {"slot", "polar", "content"}
            or (playing or opining) and p.plan.knowledge_limit is not None
            or p.operation == "opinion" and p.premises
            or p.plan.knowledge_limit is not None and query_result is not None
            and query_result.status == "Unknown"
            and p.plan.knowledge_limit.query == query_result.query
            or p.plan.clarification is not None
            or committed_reaction(p)
            or delivered_support(p)
            or p.operation in {"bind_delivered_variable", "ask_unbound_variable", "repair"}
            and any(e.kind in {"sent_turn", "human_question_grammar"} for e in p.premises)
            or p.operation == "acknowledge_report" and p.premises
            or p.operation == "explain_delivered_basis" and p.premises
            or p.operation == "verify_reported_position"
            or p.operation == "supposition" and p.premises
            or p.operation == "curiosity" and p.premises
            or p.plan.speech_act == "smalltalk" and reactive_act(
                request_text or event.meaning.text) == "smalltalk"
            or conditional_answer(p)
            or canonical_answer(p)
            or causal_answer(p)
            or p.premises and _supports(objective, p.plan.claim)
        ))
        # (a game is no answer: it competes with the honest limit instead of displacing it)
        answers = tuple(p for p in proposals if p.plan.status == "inform"
                        and p.plan.claim.clause_kind == "statement"
                        and dict(p.plan.hypothesis).get("kind") not in {"fantasy", "opinion"})
        if objective.kind in {"slot", "polar", "content"} and answers:
            # (the game, if offered, stays next to the answers)
            proposals = tuple(p for p in proposals if p in answers or dict(
                p.plan.hypothesis).get("kind") == "fantasy")
        if not proposals or event.channel_id is None or event.actor_id is None:
            return None
        review = getattr(response_decision, "selection_context", None) or {}
        ledger = await self.boredom_ledger(event)
        # (4.42, flag only) what this person is predicted to do after each kind of move, and who
        # the person is for the agent
        forecast = await self.person_forecast(event) if ledger is not None else {}
        standing = await self.person_standing(event) if ledger is not None else None
        expressed = tuple(p for p in proposals if p.plan.key in review.get("viable_plan_keys", ())
                          or (playing or opining) and p.plan.knowledge_limit is not None)
        if expressed:
            proposals = expressed
        # Scope preferences to this relationship. Neither word frequency nor
        # another participant's approval supplies a value for this choice.
        context_key = _json((objective.kind, objective.question_word))
        rows = await self._database.fetchall(
            """SELECT behavior_key, operation, outcome_type, reward, intention_id, state_json
               FROM language_meaning_intentions
               WHERE guild_id = ? AND channel_id = ? AND user_id = ?
                 AND context_key = ? AND status = 'observed' AND reward != 0
                 AND observed_at < ? AND sent_at < ?
               ORDER BY observed_at DESC LIMIT 512""",
            (
                event.guild_id,
                event.channel_id,
                event.actor_id,
                context_key,
                event.utc_timestamp.isoformat(),
                event.utc_timestamp.isoformat(),
            ),
        )
        consequences = await self._database.fetchall(
            """SELECT i.behavior_key, i.operation, c.outcome_type, i.intention_id,
                      c.observed_at AS observed_at
               FROM language_meaning_consequences c
               JOIN language_meaning_intentions i USING (intention_id)
               WHERE i.guild_id = ? AND i.channel_id = ? AND i.user_id = ?
                 AND i.context_key = ? AND c.observed_at < ? AND i.sent_at < ?
               ORDER BY c.observed_at DESC LIMIT 512""",
            (
                event.guild_id,
                event.channel_id,
                event.actor_id,
                context_key,
                event.utc_timestamp.isoformat(),
                event.utc_timestamp.isoformat(),
            ),
        )
        alternatives = []
        state = dict(state_features)
        shared = (await self._database.fetchall(
            """SELECT behavior_key, operation, outcome_type, reward, intention_id, state_json
               FROM language_meaning_intentions WHERE guild_id=? AND user_id != ?
               AND context_key=? AND status='observed' AND reward != 0
               AND observed_at < ? AND sent_at < ? ORDER BY observed_at DESC LIMIT 512""",
            (event.guild_id, event.actor_id, context_key, event.utc_timestamp.isoformat(),
             event.utc_timestamp.isoformat())) if review.get("enabled") else ())

        def relevance(row) -> float:
            previous = json.loads(row["state_json"])
            common = state.keys() & previous.keys()
            distance = sum(abs(state[key] - previous[key]) for key in common)
            return 1.0 / (1.0 + distance / len(common)) if common else 1.0

        for proposal in proposals:
            key = behavior_key(proposal)
            exact = [row for row in rows if row["behavior_key"] == key]
            samples = exact or [row for row in rows if row["operation"] == proposal.operation]
            local = bool(samples)
            if not samples:
                samples = [row for row in shared if row["behavior_key"] == key] or [
                    row for row in shared if row["operation"] == proposal.operation]
            value = (
                sum(float(row["reward"]) * relevance(row) for row in samples)
                / sum(relevance(row) for row in samples)
                if samples
                else 0.0
            )
            outcomes: dict[str, list[float]] = {}
            for row in samples:
                outcomes.setdefault(row["outcome_type"], []).append(float(row["reward"]))
            preferred = max(
                (outcome for outcome, values in outcomes.items() if sum(values) > 0),
                key=lambda outcome: sum(outcomes[outcome]) / len(samples),
                default=None,
            )
            tie = hashlib.sha256(f"{event.event_id}|{key}".encode()).hexdigest()
            tired = None
            if ledger is not None:
                # Not a draw: among equals the agent leaves what it has said most lately.
                limit = proposal.plan.knowledge_limit
                tired = ledger.reading(
                    behavior_key=key, operation=proposal.operation,
                    terms=claim_terms(proposal.plan.claim),
                    scope=limit_scope(limit.query) if limit is not None else None)
            effects = [row for row in consequences if row["behavior_key"] == key] or [
                row for row in consequences if row["operation"] == proposal.operation
            ]
            frequencies: dict[str, int] = {}
            for row in effects:
                kind = row["outcome_type"]
                frequencies[kind] = frequencies.get(kind, 0) + 1
            unclosed, unclosed_ids = (unclosed_share(effects, event.utc_timestamp)
                                      if ledger is not None else (0.0, ()))
            move = move_type(proposal.operation, dict(proposal.plan.hypothesis).get("kind"))
            seen = forecast.get(move)
            # a kind of move tested fewer than twice by this person's reactions does not act
            acts = bool(seen) and seen["checks"] >= 2
            person = ({"move": move, **({
                "checks": seen["checks"], "top": seen["top"], "trust": seen["trust"],
                "hypotheses": seen["hypotheses"], "acts": acts,
                "cost": reception_cost(move, seen["trust"]) if acts else 0.0}
                if seen else {})} if ledger is not None else None)
            alternatives.append(
                {
                    "behavior_key": key,
                    "plan_key": proposal.plan.key,
                    "operation": proposal.operation,
                    "expected_reward": value,
                    "expected_outcome": preferred,
                    "sample_count": len(samples),
                    "value_origin": "shared_operation_experience" if samples and not local
                    else "exact_experience"
                    if exact
                    else "operation_experience"
                    if samples
                    else "untrained",
                    "tie": tie,
                    **({"boredom": tired.as_data()} if tired is not None else {}),
                    "experience_ids": tuple(row["intention_id"] for row in samples),
                    **({"unclosed": unclosed, "unclosed_ids": unclosed_ids, "person": person}
                       if ledger is not None else {}),
                    "predicted_consequences": {
                        kind: count / len(effects) for kind, count in frequencies.items()
                    },
                }
            )
        if ledger is not None:
            # A question about the unknown (4.38) has no experience yet: it is taken to be
            # worth as much as the best thing the agent knows to do here (optimism about the
            # untried, not a draw). Once it has been tried, its own outcomes replace this.
            known = [a["expected_reward"] for a in alternatives if a["sample_count"]]
            for option in alternatives:
                if (option["operation"] in {"curiosity", "opinion"}
                        and not option["sample_count"] and known):
                    option["expected_reward"] = max(known)
                    option["value_origin"] = "optimistic_prior"
        draw = int(hashlib.sha256((event.event_id + ":explore").encode()).hexdigest()[:8], 16)
        # With the journal (4.33) nothing is drawn: exploring is only the cold start.
        exploring = not rows or (ledger is None and draw / 2**32 < self._exploration_rate)
        selected = max(
            range(len(alternatives)),
            key=lambda i: (
                (0.0 if exploring else alternatives[i]["expected_reward"])
                - alternatives[i].get("unclosed", 0.0)
                - (alternatives[i].get("person") or {}).get("cost", 0.0),
                proposals[i].confidence if committed_reaction(proposals[i]) else 0.0,
                # a view with a ground of its own (S) outranks a bare limit when nothing decides
                proposals[i].confidence if proposals[i].operation == "opinion" else 0.0,
                (-alternatives[i]["boredom"]["value"], alternatives[i]["behavior_key"])
                if ledger is not None else alternatives[i]["tie"],
            ),
        )
        choice = {}
        if review.get("enabled"):
            selected, choice = await self._personal_choice(
                event, proposals, alternatives, state, review, selected, ledger, standing)
            exploring = not rows
            review["choice"] = choice
        proposal, option = proposals[selected], alternatives[selected]
        intention_id = (
            "meaning-intent:"
            + hashlib.sha256(f"{event.event_id}|{proposal.plan.key}".encode()).hexdigest()[:32]
        )
        await self._database.execute(
            """INSERT OR IGNORE INTO language_meaning_intentions (
                intention_id, parent_intent_id, prompt_event_id, guild_id, channel_id,
                user_id, context_key, behavior_key, operation, plan_key, plan_json,
                expected_reward, expected_outcome, sample_count, selection_json,
                state_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                intention_id,
                parent_intent_id,
                event.event_id,
                event.guild_id,
                event.channel_id,
                event.actor_id,
                context_key,
                option["behavior_key"],
                proposal.operation,
                proposal.plan.key,
                _json(identity_graph(proposal.plan)),
                option["expected_reward"],
                option["expected_outcome"],
                option["sample_count"],
                _json({"exploring": exploring, "alternatives": alternatives, "choice": choice,
                       "response_decision": response_behavior(response_decision)}),
                _json(state),
                event.utc_timestamp.isoformat(),
            ),
        )
        return MeaningSelection(
            intention_id,
            proposal,
            option["expected_reward"],
            option["expected_outcome"],
            option["sample_count"],
            tuple(alternatives),
            exploring,
            choice,
        )

    async def _personal_choice(self, event, proposals, alternatives, state, review, score_first,
                               ledger=None, standing=None):
        """Proportional contributions rank meanings already admitted by evidence."""
        from debilgi.language.content import content_lemmas
        from debilgi.language.dialogue_acts import _STUDY_PREDICATES
        from debilgi.personality.genome import DEBILGI_GENOME

        history = await self._database.fetchall(
            """SELECT intention_id, behavior_key, response_text, operation, prompt_event_id
               FROM language_meaning_intentions WHERE guild_id=? AND channel_id=? AND user_id=?
               AND sent_at < ? AND sent_at >= ? AND response_message_id IS NOT NULL
               ORDER BY sent_at DESC LIMIT 16""",
            (event.guild_id, event.channel_id, event.actor_id, event.utc_timestamp.isoformat(),
             (event.utc_timestamp - timedelta(minutes=10)).isoformat()))
        memories = await self._database.fetchall(
            """SELECT d.source_id, d.content FROM language_discourse_turns d
               JOIN events e ON e.event_id=d.source_id
               WHERE d.guild_id=? AND d.channel_id=? AND d.actor_id=? AND d.observed_at < ?
               AND COALESCE(json_extract(e.payload_json, '$.author_is_bot'),0)=0
               ORDER BY d.observed_at DESC LIMIT 32""",
            (event.guild_id, event.channel_id, event.actor_id, event.utc_timestamp.isoformat()))
        traits = {**DEBILGI_GENOME.traits(), **review.get("personality", {})}
        affect = {**review.get("affect", {}), **state}
        valence = 0.72 * affect.get("emotion_valence", 0) + 0.28 * affect.get("mood_valence", 0)
        # Irritation (flag only) reads like a negative valence towards this person, with the
        # same coefficients as valence itself; it also makes a retort worth something.
        irritation = affect.get("irritation", 0.0)
        valence -= irritation
        if standing is not None:
            # P (4.42): the warmth of this person's reactions to the agent's actions colours the
            # valence the same way irritation does; irritation over T_S counts when it outlasts
            # the minute's
            valence += standing["warmth"] - max(0.0, standing["irritation"] - irritation)
        relation = review.get("relation", {})
        drives = {
            "curiosity": traits["curiosity"] - 0.5 + affect.get("state_interest", 0) * 0.3,
            "affiliation": traits.get("warmth", traits["sociability"]) - 0.5 + valence * 0.5,
            "caution": traits["suspicion"] - 0.5 - valence * 0.5
                       - (relation.get("trust", 0.5) - 0.5) * 0.2,
            "playfulness": traits.get("playfulness", traits["chaos"]) - 0.5 - 0.5 * irritation,
            **({"defense": irritation} if irritation else {}),
        }
        features = []
        for i, proposal in enumerate(proposals):
            plan, option = proposal.plan, alternatives[i]
            terms = set(content_lemmas(" ".join(
                (plan.claim.predicate.lemma, *(d.argument.head.lemma
                                              for d in plan.claim.dependencies),
                 *(value for _, value, _ in plan.known_slots)))))
            sources = [m["source_id"] for m in memories
                       if terms & set(content_lemmas(m["content"]))]
            memory_match = sum(len(terms & set(content_lemmas(m["content"])))
                               / max(1, len(terms)) for m in memories)
            repetitions = [h["intention_id"] for h in history
                           if h["behavior_key"] == option["behavior_key"]]
            act = plan.claim.predicate.lemma
            features.append({
                "memory": memory_match, "memory_sources": sources,
                "repeat": len(repetitions) if ledger is None else 0,
                "repeat_sources": repetitions,
                **({"boredom": option["boredom"], "unclosed": option["unclosed"],
                    "person": (option.get("person") or {}).get("cost", 0.0)}
                   if ledger is not None else {}),
                "experience": option["expected_reward"] if option["sample_count"]
                or option["value_origin"] == "optimistic_prior" else 0.0,
                "specificity": len(plan.known_slots) + len(plan.claim.dependencies),
                "curiosity": int(plan.claim.clause_kind == "question"
                                 or proposal.operation == "curiosity"
                                 or act in _STUDY_PREDICATES | {
                                     "slot_question", "reciprocal_question", "check_condition"}),
                "affiliation": int(act in {"agree", "empathize", "return_greeting"}),
                # (4.45) with a view of its own to give, not knowing is not the cautious choice
                "caution": int((bool(plan.knowledge_limit or plan.clarification)
                                and not any(p.operation == "opinion" for p in proposals))
                               or act in {"acknowledge", "check_condition"}),
                # Play is a taste of character (genome playfulness).
                "playfulness": int(act in {"mirror_amusement", "retort"}
                                   or dict(plan.hypothesis).get("kind") == "fantasy"),
                "coalition": plan.claim.confidence,
                **({"defense": int(act == "retort")} if irritation else {}),
            })
        contributions = [{
            "experience": (f["experience"] - f.get("unclosed", 0.0)) * 3.5,
            "memory": f["memory"] / (1 + f["memory"]) * traits["memory_attachment"] * .2
                      - f["repeat"] * 1.5,
            **({"boredom": -BOREDOM_PENALTY * f["boredom"]["value"],
                "person": -f["person"] * 3.5} if ledger is not None else {}),
            "grounding": f["specificity"] / (1 + f["specificity"]) * .1,
            "taste": sum(f[drive] * strength for drive, strength in drives.items()),
            "coalition": f["coalition"] * .25,
        } for f in features]
        totals = [sum(c.values()) for c in contributions]
        selected = max(range(len(proposals)), key=lambda i: totals[i])
        # Describe the largest actual contrast; all contributions remain visible.
        other = next((i for i in range(len(proposals)) if i != selected), selected)
        reason = max(contributions[selected], key=lambda k:
                     contributions[selected][k] - contributions[other][k])
        reasons = [{"source": reason, "criterion": "proportional_contributions",
                    "contributions": contributions, "totals": totals,
                    "source_ids": list(dict.fromkeys(
                        [h["intention_id"] for h in history] if reason == "memory" else
                        [e for a in alternatives for e in (
                            *a["experience_ids"], *a.get("unclosed_ids", ()))]
                        if reason == "experience" else
                        [e for p in proposals for e in p.plan.claim.evidence]))}]
        return selected, {
            "policy": "personal_preferences", "considered": len(proposals),
            "score_first": score_first, "selected": selected,
            "not_first_by_score": selected != score_first, "decisive_source": reason,
            "reasons": reasons, "features": features, "drives": drives,
            "contributions": contributions, "totals": totals,
            "state": affect, "personality": traits,
            **({"standing": standing} if standing is not None else {}),
            "memory_scope": [event.guild_id, event.channel_id, event.actor_id],
            "explanation": "взвешены память, опыт, активация и текущее состояние",
        }

    async def finalize_selection(self, selection, proposal, *, decision=None, pattern_id=None,
                                 basis=None):
        """The arbiter commits the actually realizable meaning before delivery.

        Preserve the intention identity used by candidate metadata/receipts;
        feedback must train the delivered plan, not a provisional competitor.
        """
        option = next(a for a in selection.alternatives if a["plan_key"] == proposal.plan.key)
        choice = {**selection.choice, "final_plan_key": proposal.plan.key,
                  "provisional_plan_key": selection.proposal.plan.key}
        basis = basis if basis is not None else delivered_basis(proposal.premises)
        passed_over = {a["behavior_key"]: list(a["unclosed_ids"]) for a in selection.alternatives
                       if a["plan_key"] != proposal.plan.key and a.get("unclosed_ids")}
        if "unclosed" in option and (option["unclosed_ids"] or option["experience_ids"]
                                     or passed_over):
            # (4.41, flag only) the agent's own earlier turns whose outcomes this choice weighed:
            # of the chosen action, and of the actions it was preferred to
            basis = {**basis, "earlier_outcomes": {
                "left_open": list(option["unclosed_ids"]),
                "rewarded": list(option["experience_ids"]), "unclosed": option["unclosed"],
                **({"passed_over_left_open": passed_over} if passed_over else {})}}
        if selection.choice.get("standing"):
            # (4.42, flag only) who this person is for the agent when the choice was made: a
            # bias towards the liked or the disliked is visible here, not hidden in a weight
            basis = {**basis, "person_standing": selection.choice["standing"]}
        person = option.get("person") or {}
        if person.get("hypotheses"):
            # (4.42, flag only) the hypotheses about THIS person that this choice weighed: their
            # ids, evidence and the forecast; and, when they decided it, whom they passed over
            others = {a["behavior_key"]: a["person"]["hypotheses"] for a in selection.alternatives
                      if a["plan_key"] != proposal.plan.key
                      and (a.get("person") or {}).get("cost")}
            basis = {**basis, "person_hypotheses": {
                "move": person["move"], "forecast": person["top"], "checks": person["checks"],
                "acts": person["acts"], "cost": person["cost"],
                "hypotheses": person["hypotheses"],
                **({"passed_over_costly": others} if others else {})}}
        elif any((a.get("person") or {}).get("cost") for a in selection.alternatives):
            basis = {**basis, "person_hypotheses": {"passed_over_costly": {
                a["behavior_key"]: a["person"]["hypotheses"] for a in selection.alternatives
                if (a.get("person") or {}).get("cost")}}}
        await self._database.execute(
            """UPDATE language_meaning_intentions SET behavior_key=?, operation=?,
               plan_key=?, plan_json=?, expected_reward=?, expected_outcome=?, sample_count=?,
               selection_json=? WHERE intention_id=? AND response_message_id IS NULL""",
            (option["behavior_key"], proposal.operation, proposal.plan.key,
             _json(identity_graph(proposal.plan)), option["expected_reward"],
             option["expected_outcome"],
             option["sample_count"], _json({"alternatives": selection.alternatives,
                                          "choice": choice,
                                          "response_decision": response_behavior(decision),
                                          "exchange_pattern_id": pattern_id,
                                          "basis": basis}),
             selection.intention_id))
        return replace(selection, proposal=proposal, expected_reward=option["expected_reward"],
                       expected_outcome=option["expected_outcome"],
                       sample_count=option["sample_count"], choice=choice)

    async def adopt_hypothesis(
        self, selection, plan: dict, hypothesis_id: str, plan_key: str | None = None,
        *, decision=None, basis=None
    ):
        """Remember a night hypothesis that won by day as the plan actually delivered.

        This is experience of the bot's own action, like any intention: it never enters
        ``meaning_claims`` and does not confirm the hypothesis (only a person's feedback
        on its receipt does).
        """
        from debilgi.language.communication import MessagePlan, ReaderGoal

        executed = DerivedPlan(
            MessagePlan(ReaderGoal(**plan["goal"]), thought_from_data(plan["claim"]),
                        speech_act=plan.get("speech_act")),
            "sleep_hypothesis", ())
        key = plan_key or executed.plan.key
        choice = {**selection.choice, "sleep_hypothesis_id": hypothesis_id,
                  "final_plan_key": key, "provisional_plan_key": selection.proposal.plan.key}
        await self._database.execute(
            """UPDATE language_meaning_intentions SET behavior_key=?, operation=?,
               plan_key=?, plan_json=?, expected_reward=0, expected_outcome=NULL, sample_count=0,
               selection_json=? WHERE intention_id=? AND response_message_id IS NULL""",
            (behavior_key(executed), "sleep_hypothesis", key, _json(plan),
             _json({"alternatives": selection.alternatives, "choice": choice,
                    "response_decision": response_behavior(decision),
                    "basis": basis if basis is not None else delivered_basis()}),
             selection.intention_id))
        return replace(selection, expected_reward=0.0, expected_outcome=None, sample_count=0,
                       choice=choice)

    async def record_sent(
        self,
        prompt: Event,
        *,
        intention_id: str,
        response_message_id: int,
        created_at: datetime,
        text: str | None = None,
    ) -> bool:
        # This describes a real operation only AFTER Discord confirms sending.
        # Naming the executed action does not give it a positive value/desire.
        action = ClauseThought(
            ConceptLexeme("отвечать", "отвечать", "verb"),
            (
                SemanticDependency("subject", ConceptArgument(ConceptLexeme("я", "я", "pronoun"))),
                SemanticDependency(
                    "indirect_object",
                    ConceptArgument(
                        ConceptLexeme("ты", "ты", "pronoun"), grammatical_case="dative"
                    ),
                ),
            ),
            confidence=1.0,
        )
        cursor = await self._database.execute(
            """UPDATE language_meaning_intentions
               SET status = 'sent', response_message_id = ?, sent_at = ?, action_json = ?,
                   response_text = ?
               WHERE intention_id = ? AND prompt_event_id = ? AND guild_id = ?
                 AND channel_id = ? AND user_id = ? AND status = 'prepared'""",
            (
                response_message_id,
                created_at.isoformat(),
                _json(asdict(action)),
                text,
                intention_id,
                prompt.event_id,
                prompt.guild_id,
                prompt.channel_id,
                prompt.actor_id,
            ),
        )
        sent = cursor.rowcount == 1
        if sent:
            row = await self._database.fetchone(
                "SELECT plan_json,operation,selection_json FROM language_meaning_intentions "
                "WHERE intention_id=?",
                (intention_id,),
            )
            plan = json.loads(row[0]) if row else {}
            control = plan.get("deliberation", {})
            control = control.get("control", control)
            if (row and text and text.rstrip().endswith("?") and control.get("enabled")
                    and row["operation"] in {
                        "retrieve_proposition", "retrieve_self_claim", "retrieve_delivered_act"}
                    and plan.get("claim", {}).get("clause_kind") == "statement"
                    and not plan.get("knowledge_limit")):
                # Persist what was actually sent, not the failed assertion.
                # Matching the original proof body also works after restart;
                # there is no separate in-memory execution ledger.
                from dataclasses import replace

                from debilgi.language.communication import MessagePlan, ReaderGoal, surface_issues

                question = replace(thought_from_data(plan["claim"]),
                                   clause_kind="question", focus_role=None, confidence=.49)
                if question.evidence and not surface_issues(question, text):
                    goal = ReaderGoal(**{**plan["goal"], "kind": "respond", "question_word": None})
                    executed = MessagePlan(goal, question, speech_act="verify_known_content")
                    proposal = DerivedPlan(executed, "verify_known_content", ())
                    selection = json.loads(row["selection_json"])
                    selection["executed_reconsideration"] = {
                        "prepared_operation": row["operation"], "source_ids": question.evidence,
                        "operation": proposal.operation, "plan_key": executed.key}
                    plan = identity_graph(executed)
                    await self._database.execute(
                        "UPDATE language_meaning_intentions SET plan_json=?,plan_key=?,"
                        "operation=?,behavior_key=?,selection_json=?,expected_reward=0,"
                        "expected_outcome=NULL,sample_count=0 WHERE intention_id=?",
                        (_json(plan), executed.key, proposal.operation, behavior_key(proposal),
                         _json(selection), intention_id))
            if row["operation"] == "opinion":
                await self._hold_position(prompt, intention_id, plan, response_message_id,
                                          created_at)
            if (plan.get("hypothesis") and row["operation"] != "sleep_hypothesis"
                    and dict(plan["hypothesis"]).get("kind") not in {"fantasy", "opinion"}):
                # (make-believe is no hypothesis of the world: nothing to confirm by a «да»)
                from debilgi.cognition.continuous import ContinuousCognitionService

                await ContinuousCognitionService(self._database).record_response_hypothesis(
                    prompt, plan["hypothesis"], plan["claim"]["confidence"], response_message_id)
        return sent

    async def _hold_position(self, prompt: Event, intention_id: str, plan: dict,
                             response_message_id: int, created_at: datetime) -> None:
        """A delivered opinion is the agent's stance (4.45, flag only): written to the existing
        `language_positions` (+ `language_expressed_positions` for the reactions to it), and only
        when sent — thinking and sleep never write it. A flip (`revised`) replaces the held
        stance; otherwise the held one stays. Never a claim, evidence or hypothesis."""
        from debilgi.language.position_reasoning import (
            PositionDecision,
            PositionRepository,
            _position_key,
        )

        view = dict(plan.get("hypothesis") or {})
        if view.get("kind") != "opinion" or not view.get("topic"):
            return
        key = _position_key(view["topic"], view["quality"])
        score = float(view["score"])
        confidence = min(1.0, max(0.0, abs(score)))
        repository = PositionRepository(self._database)
        await repository.record_expressed(
            PositionDecision(key, view["topic"], view["quality"], view["stance"], score,
                             confidence, tuple(json.loads(view.get("ground") or "[]")),
                             "opinion", "answer", "state_position", "direct"),
            guild_id=prompt.guild_id, source_event_id=prompt.event_id, candidate_id=intention_id,
            response_message_id=response_message_id, observed_at=created_at)
        if view.get("revised"):
            await self._database.execute(
                "UPDATE language_positions SET stance=?, score=?, confidence=? "
                "WHERE guild_id=? AND position_key=?",
                (view["stance"], score, confidence, prompt.guild_id, key))

    async def observe(self, event: Event) -> bool:
        if event.payload.get("author_is_bot") or event.actor_id is None:
            return False
        feedback = DEFAULT_FEEDBACK_ATTRIBUTOR.attribute(event)
        # No nearest-turn attribution, continuation reward or silence penalty.
        # An explicit target is needed to claim a consequence of this action.
        if feedback.target_message_id is None:
            return False
        if event.event_type is EventType.REACTION_REMOVED:
            # Removing a reaction does not mean endorsing its opposite.
            return False
        prior = await self._database.fetchone(
            """SELECT intention_id, prompt_event_id, plan_json, response_text, operation
               FROM language_meaning_intentions
               WHERE guild_id = ? AND channel_id = ? AND user_id = ?
                 AND response_message_id = ? AND status = 'sent'
                 AND sent_at < ? AND sent_at >= ?""",
            (
                event.guild_id,
                event.channel_id,
                event.actor_id,
                feedback.target_message_id,
                event.utc_timestamp.isoformat(),
                (event.utc_timestamp - self._feedback_window).isoformat(),
            ),
        )
        if prior is None:
            return False
        plan = json.loads(prior["plan_json"])
        claim = plan["claim"]
        outcome = feedback.kind
        effect = None
        reward = feedback.reward if feedback.is_learnable() else None
        if event.event_type in {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED}:
            content = str(event.payload.get("content", ""))
            words = set(tokenize(content.casefold()))
            # "Так да или нет?" is a demand for an answer, not approval merely
            # because the affirmative token occurs in it.
            if (("?" in content or words & {"или", "либо"})
                    and feedback.kind not in {"explicit_rejection", "contextual_correction"}):
                reward = None
                outcome = "follow_up_question"
        if (reward is None and outcome != "follow_up_question"
                and event.event_type in {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED}
                and await self._presses_demand(event, prior)):
            outcome = "follow_up_question"
        elif (outcome == "follow_up_question" and feedback.kind not in {
                "explicit_rejection", "contextual_correction"}
                and event.event_type in {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED}
                and not set(tokenize(str(event.payload.get("content", "")).casefold()))
                & {"или", "либо"} and await self._asks_on(event, prior)):
            # (4.42, flag only) a new question on the topic is the person going on, not the same
            # demand left open: it is not counted as «unclosed»
            outcome = "conversation_continued"
        if claim["clause_kind"] == "question" and event.event_type in {
            EventType.MESSAGE_CREATED,
            EventType.MESSAGE_REPLIED,
        }:
            words = set(tokenize(str(event.payload.get("content", "")).casefold()))
            # A yes/no answer supplies information; it does not praise or
            # punish asking the question. Refusal labels are also ambiguous.
            if feedback.kind == "refusal" or words <= AFFIRMATION | REFUSAL:
                reward = None
                outcome = "question_answer"
            if prior["response_text"]:
                exchange = resolve_dialogue_exchange(
                    str(event.payload.get("content", "")),
                    (prior["response_text"],),
                )
                if exchange is not None and exchange.status == "matched":
                    outcome = "requested_information_received"
                    effect = _json(identity_graph(exchange.bound_proposition))
                    reward = None
            clarification = plan.get("clarification")
            if clarification and "?" not in str(event.payload.get("content", "")):
                from debilgi.language.content import content_lemmas

                supplied = set(content_lemmas(str(event.payload.get("content", ""))))
                bound = supplied.intersection(clarification["alternatives"])
                if len(bound) == 1:
                    outcome, reward = "requested_information_received", None
                    effect = _json({"kind": "reference_binding",
                                    "ambiguous_slot": clarification["ambiguous_slot"],
                                    "value": next(iter(bound)), "source_event_id": event.event_id})
        if (event.event_type in {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED}
                and outcome in {"conversation_continued", "requested_information_received"}):
            outcome = await self._observe_answer(event, prior, plan, outcome)  # (4.44, flag only)
        await self._database.execute(
            """INSERT OR IGNORE INTO language_meaning_consequences (
                intention_id, event_id, outcome_type, semantic_effect_json,
                external_value, observed_at
            ) VALUES (?, ?, ?, ?, ?, ?)""",
            (
                prior["intention_id"],
                event.event_id,
                outcome,
                effect,
                reward,
                event.utc_timestamp.isoformat(),
            ),
        )
        from debilgi.language.response_trace import ResponseTraceRepository

        await ResponseTraceRepository(self._database).step(
            str(prior["prompt_event_id"]), stage="feedback", component="meaning_consequences",
            status="observed", summary="attributed consequence; delivery is recorded separately",
            details={"feedback_event_id": event.event_id, "intention_id": prior["intention_id"],
                     "target_message_id": feedback.target_message_id, "outcome": outcome,
                     "external_value": reward, "semantic_effect": effect},
            created_at=event.utc_timestamp,
        )
        await self.observe_person(event, prior, outcome)  # (4.42, flag only)
        # Descriptive partner hypotheses share the attribution boundary, but
        # continuation supplies no reward and cannot create a learned goal.
        from debilgi.language.interlocutor_intent import InterlocutorIntentRepository

        await InterlocutorIntentRepository(self._database).observe_live(
            event, prior, outcome, reward,
        )
        if reward is None:
            return False
        cursor = await self._database.execute(
            """UPDATE language_meaning_intentions
               SET status = 'observed', feedback_event_id = ?, outcome_type = ?,
                   reward = ?, observed_at = ?
               WHERE guild_id = ? AND channel_id = ? AND user_id = ?
                 AND response_message_id = ? AND status = 'sent'
                 AND sent_at < ? AND sent_at >= ?""",
            (
                event.event_id,
                outcome,
                reward,
                event.utc_timestamp.isoformat(),
                event.guild_id,
                event.channel_id,
                event.actor_id,
                feedback.target_message_id,
                event.utc_timestamp.isoformat(),
                (event.utc_timestamp - self._feedback_window).isoformat(),
            ),
        )
        return cursor.rowcount == 1

    async def sympathy(self, event: Event, topics, since: str | None = None) -> Sympathy:
        """S(topic): what people's reactions to the agent's own claims about the topic, and the
        agent's own deeds, say about how it fares with it (4.45, flag only).

        Reactions: a delivered turn that *said* something about the topic (a claim, not a
        limit, not a question) and the person's reaction to it, as the reaction class of H
        (`reaction_class`; warmth of `_WARMTH`, valence 2w-1), **relative to the same person's
        average reaction** to whatever the agent said (otherwise every topic spoken of while the
        talk went on would be liked). One person is one voice: v = Σ a·r / (Σ a + 1) over his
        reactions weighted by a = 2^(-age/T_S); the voice counts as much as the person's warmth
        for the agent (2·`bot_action_response`: a neutral person 1). Deeds: what the agent keeps
        doing (completed own activities) is one voice of its own, Σ a / (Σ a + 1). S = tanh of
        the sum. With no reaction and no deed S is 0 with no mass: no attitude exists. Turns that
        only admit not knowing (a limit, a question) are no experience of the topic. Only this
        server's journals are read; the seed corpus is not (it has no turns of the agent)."""
        from debilgi.cognition.models import EntityType
        from debilgi.config import thought_content_enabled

        wanted = frozenset(t for t in topics if t)
        if not thought_content_enabled() or not wanted:
            return Sympathy()
        now = event.utc_timestamp
        half_life = social_half_life()
        horizon = now.isoformat()
        reactions = {}
        rows = await self._database.fetchall(
            """SELECT i.intention_id, i.user_id, i.operation, i.plan_json, c.outcome_type,
                      c.event_id, c.observed_at
               FROM language_meaning_consequences c
               JOIN language_meaning_intentions i USING (intention_id)
               WHERE i.guild_id = ? AND c.observed_at < ? AND i.response_message_id IS NOT NULL
               ORDER BY c.observed_at DESC LIMIT 4096""", (event.guild_id, horizon))
        if rows:
            seen = await self._database.fetchall(
                "SELECT source_event_id, details_json FROM continuous_cognition_evidence "
                "WHERE evidence_id LIKE 'person-evidence:%' AND relation='supports'")
            reactions = {r["source_event_id"]: json.loads(r["details_json"]).get("reaction")
                         for r in seen}

        def valence(row) -> float:
            reaction = reactions.get(row["event_id"]) or reaction_class(
                "", row["outcome_type"], False)
            return 2.0 * _WARMTH.get(reaction, 0.5) - 1.0

        weight = {id(r): decay_weight(_age({"observed_at": r["observed_at"]}, now), half_life)
                  for r in rows}
        average: dict[int, float] = {}
        for person in {r["user_id"] for r in rows}:
            mine = [r for r in rows if r["user_id"] == person]
            mass = sum(weight[id(r)] for r in mine)
            average[person] = sum(weight[id(r)] * valence(r) for r in mine) / mass if mass else 0.0
        votes: dict[int, list[tuple[float, float]]] = {}
        ids, newest = [], None
        for row in rows:
            plan = json.loads(row["plan_json"])
            if (not plan.get("claim") or plan.get("knowledge_limit")
                    or plan["claim"].get("clause_kind") == "question"
                    or (row["operation"] or "") in _COUNTER_OPERATIONS
                    or (row["operation"] or "").startswith("clarify_")
                    or row["operation"] == "acknowledge_report"):
                continue
            if since is not None and row["observed_at"] <= since:
                continue
            if not any(_same_topic(wanted, term) for term in _plan_terms(row["plan_json"])):
                continue
            votes.setdefault(row["user_id"], []).append(
                (weight[id(row)], valence(row) - average[row["user_id"]]))
            ids.append(row["intention_id"])
            newest = max(newest or "", row["observed_at"])
        total, weights = 0.0, []
        for person, items in votes.items():
            mass = sum(a for a, _ in items)
            voice = sum(a * r for a, r in items) / (mass + 1.0)
            model = (await self._cognition.entity_model(event.guild_id, EntityType.USER, person)
                     if self._cognition is not None else None)
            warmth = 2.0 * (model.bot_action_response if model else 0.5)
            weights.append(warmth)
            total += warmth * voice
        deeds = await self._database.fetchall(
            """SELECT a.activity_id, a.started_at, c.predicate, c.claim_id FROM agent_activities a
               JOIN meaning_claims c ON c.claim_id = a.claim_id
               WHERE c.guild_id = ? AND a.status = 'completed' AND a.started_at < ?
               ORDER BY a.started_at DESC LIMIT 1024""", (event.guild_id, horizon))
        habit = [d for d in deeds if _same_topic(wanted, d["predicate"])
                 and (since is None or d["started_at"] > since)]
        if habit:
            mass = sum(decay_weight(_age({"observed_at": d["started_at"]}, now), half_life)
                       for d in habit)
            total += mass / (mass + 1.0)
            ids.extend(d["activity_id"] for d in habit[:4])
            newest = max(newest or "", habit[0]["started_at"])
        return Sympathy(math.tanh(total), len(votes), sum(len(v) for v in votes.values()),
                        len(habit), tuple(ids), tuple(weights), newest,
                        habit[0]["claim_id"] if habit else None)

    async def unclosed_by_operation(self, event: Event, objective) -> dict[str, float]:
        """Operation -> share of its earlier outcomes (this person, this context) that left the
        demand open (4.41, flag only). Read before the proposals are completed, so that an
        operation that stably does not close the demand meets alternatives."""
        from debilgi.config import thought_content_enabled

        if not thought_content_enabled() or event.actor_id is None or event.channel_id is None:
            return {}
        rows = await self._database.fetchall(
            """SELECT i.operation, c.outcome_type, i.intention_id, c.observed_at AS observed_at
               FROM language_meaning_consequences c
               JOIN language_meaning_intentions i USING (intention_id)
               WHERE i.guild_id = ? AND i.channel_id = ? AND i.user_id = ?
                 AND i.context_key = ? AND c.observed_at < ? AND i.sent_at < ?
               ORDER BY c.observed_at DESC LIMIT 512""",
            (event.guild_id, event.channel_id, event.actor_id,
             _json((objective.kind, objective.question_word)), event.utc_timestamp.isoformat(),
             event.utc_timestamp.isoformat()))
        result = {}
        for operation in {row["operation"] for row in rows}:
            share, _ = unclosed_share([r for r in rows if r["operation"] == operation],
                                      event.utc_timestamp)
            if share:
                result[operation] = share
        return result

    async def unwelcome_by_operation(self, event: Event) -> dict[str, float]:
        """Operation -> what this person's reactions predict it costs (4.42, flag only): the
        kinds of move that the person, tested twice or more, meets with laughter (for a serious
        move), hostility or rejection. Like `unclosed_by_operation`, it opens alternatives."""
        forecast = await self.person_forecast(event)
        costly = {move: reception_cost(move, entry["trust"]) for move, entry in forecast.items()
                  if entry["checks"] >= 2}
        return {move: cost for move, cost in costly.items() if cost}

    async def person_forecast(self, event: Event) -> dict[str, dict]:
        """Move kind -> what THIS person is predicted to do after it (4.42, flag only).

        Read from the hypotheses stored about this person on this server and nowhere else: what
        another person did to the agent, or what happened on another server, is not read.
        `checks` is the number of the person's reactions that tested the kind of move (a kind
        with fewer than two does not act); `trust[r]` is the age-weighted confidence in «r»."""
        from debilgi.config import thought_content_enabled

        if not thought_content_enabled() or event.actor_id is None:
            return {}
        rows = await self._database.fetchall(
            """SELECT hypothesis_id, predicate_key, object_key, confidence, positive_evidence,
                      negative_evidence FROM continuous_cognition_hypotheses
               WHERE guild_id=? AND subject_key=? AND signature LIKE 'person:%'
                 AND status != 'rejected'""",
            (event.guild_id, f"discord:user:{event.actor_id}"))
        forecast: dict[str, dict] = {}
        for row in rows:
            move = str(row["predicate_key"]).removeprefix("reacts_to:")
            entry = forecast.setdefault(move, {"checks": 0, "trust": {}, "hypotheses": []})
            entry["checks"] += int(row["positive_evidence"])
            entry["trust"][row["object_key"]] = float(row["confidence"])
            entry["hypotheses"].append({
                "id": row["hypothesis_id"], "reaction": row["object_key"],
                "supports": int(row["positive_evidence"]),
                "contradicts": int(row["negative_evidence"]),
                "confidence": round(float(row["confidence"]), 4)})
        for entry in forecast.values():
            entry["top"] = max(entry["trust"], key=lambda r: (entry["trust"][r], r))
        return forecast

    async def person_standing(self, event: Event) -> dict | None:
        """P: who this person is for the agent (4.42, flag only) — the warmth of the person's
        reactions to the agent's actions (cognition_entity_models.bot_action_response, 0..1 around
        0.5) and the irritation with the horizon T_S, not of the minute."""
        from debilgi.cognition.affect import irritation_reading
        from debilgi.cognition.models import EntityType
        from debilgi.config import thought_content_enabled

        if not thought_content_enabled() or event.actor_id is None or self._cognition is None:
            return None
        model = await self._cognition.entity_model(event.guild_id, EntityType.USER, event.actor_id)
        reading = await irritation_reading(self._database, event, event.meaning.text,
                                           half_life=social_half_life())
        from debilgi.cognition import chat_norms

        # C (4.43): what this chat showed about the words of this turn — norm ids and how many
        # people, never who; the norms already acted through the reading above and play_barred
        norms = await chat_norms.load(self._database, event.guild_id)
        about = norms.about(chat_norms._lemmas(event.meaning.text))
        return {"warmth": 2.0 * (model.bot_action_response if model else 0.5) - 1.0,
                "irritation": reading.value if reading is not None else 0.0,
                "observations": model.observations if model else 0,
                "source": "cognition_entity_models.bot_action_response; irritation over T_S",
                **({"chat_norms": about} if about else {})}

    async def observe_person(self, event: Event, prior, outcome: str) -> str | None:
        """The person's reaction to the agent's move tests the hypotheses about that person
        (4.42, flag only): a hypothesis «after a move of this kind: r» gains support when the
        reaction is r, and is contradicted when it is another one. Nothing here is a claim."""
        from debilgi.config import thought_content_enabled

        if not thought_content_enabled() or event.actor_id is None:
            return None
        plan = json.loads(prior["plan_json"])
        move = move_type(prior["operation"], dict(plan.get("hypothesis") or {}).get("kind"))
        text = str(event.payload.get("content", ""))
        hostile = False
        if event.event_type in {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED}:
            reading = await self.irritation(event, text)
            hostile = bool(reading is not None and reading.hostile)
        reaction = reaction_class(text, outcome, hostile)
        subject = f"discord:user:{event.actor_id}"
        now = event.utc_timestamp
        signature = f"{PERSON_SIGNATURE}{event.actor_id}:{move}:{reaction}"
        hypothesis_id = "person-hypothesis:" + hashlib.sha256(signature.encode()).hexdigest()[:32]
        await self._database.execute(
            """INSERT OR IGNORE INTO continuous_cognition_hypotheses
               (hypothesis_id, guild_id, signature, subject_key, predicate_key, object_key,
                roles_json, epistemic_status, confidence, positive_evidence, negative_evidence,
                status, first_seen_at, last_reviewed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'hypothesis', 0.0, 0, 0, 'working', ?, ?)""",
            (hypothesis_id, event.guild_id, signature, subject, f"reacts_to:{move}", reaction,
             _json({"kind": "person_reaction", "move": move, "reaction": reaction,
                    "actor_id": event.actor_id}), now.isoformat(), now.isoformat()))
        rows = await self._database.fetchall(
            """SELECT hypothesis_id, object_key FROM continuous_cognition_hypotheses
               WHERE guild_id=? AND subject_key=? AND predicate_key=?
                 AND signature LIKE 'person:%'""",
            (event.guild_id, subject, f"reacts_to:{move}"))
        for row in rows:
            relation = "supports" if row["object_key"] == reaction else "contradicts"
            await self._database.execute(
                """INSERT OR IGNORE INTO continuous_cognition_evidence
                   (evidence_id, hypothesis_id, source_event_id, relation, confidence, details_json,
                    observed_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (f"person-evidence:{row['hypothesis_id']}:{event.event_id}:{relation}",
                 row["hypothesis_id"], event.event_id, relation, 1.0,
                 _json({"intention_id": prior["intention_id"], "move": move,
                        "reaction": reaction}), now.isoformat()))
        await reweigh_person(self._database, [r["hypothesis_id"] for r in rows], now)
        if self._cognition is not None:
            await self._cognition.record_bot_action_response(
                event.guild_id, event.actor_id, _WARMTH[reaction], WARMTH_RATE)
        return reaction

    async def _observe_answer(self, event: Event, prior, plan: dict, outcome: str) -> str:
        """The person's reply to a question the agent asked (4.44, flag only): answered (the reply
        is a statement about what the question was about), evaded (it is not), or — checked by
        the reader of hostility in `reaction_class` — hostile. An answer is written as the
        person's position (`meaning_positions`, scope «report»): what this person said, keyed by
        the topic; never a claim of the agent and never a fact of the world.
        """
        from debilgi.config import thought_content_enabled

        operation = prior["operation"] or ""
        if not thought_content_enabled() or not (
                operation in _COUNTER_OPERATIONS or operation.startswith("clarify_")):
            return outcome
        from debilgi.language.dialogue_learning import answered_topics

        text = str(event.payload.get("content", ""))
        topics = answered_topics(text, plan)
        reading = await self.irritation(event, text)
        if not topics or (reading is not None and reading.hostile):
            return "evaded" if outcome == "conversation_continued" else outcome
        for topic in topics:
            await self._database.execute(
                "INSERT OR REPLACE INTO meaning_positions VALUES (?,?,?,?,?,?,?,?)",
                (event.guild_id, REPORT_KEY + topic, event.actor_id, REPORT_SCOPE, "*", 1,
                 event.utc_timestamp.isoformat(), event.event_id))
        return "answered" if outcome == "conversation_continued" else outcome

    async def _asks_on(self, event: Event, prior) -> bool:
        from debilgi.config import thought_content_enabled

        return thought_content_enabled() and not await self._presses_demand(event, prior)

    async def _presses_demand(self, event: Event, prior) -> bool:
        """The reply to the agent's turn is the same demand again (4.41, flag only).

        A repeat of the prompt (the same content lemmas) or a repair of the last turn
        («почему», «какое», «так что», «не понял») leaves the demand open, with or without a «?».
        """
        import re

        from debilgi.config import thought_content_enabled

        if not thought_content_enabled():
            return False
        from debilgi.language.content import content_lemmas
        from debilgi.language.dialogue_acts import analyze_turn_form

        content = re.sub(r"<@!?\d+>", " ", str(event.payload.get("content", "")))
        form = analyze_turn_form(content, addressed=True, follows_bot=True,
                                 previous_bot_text=prior["response_text"] or "")
        if form.kind == "repair":
            return True
        from debilgi.language.structured_utterance import analyze_structured_utterance

        structure = analyze_structured_utterance(content)
        if (any(t.semantic_role == "question_target" for t in structure.tokens)
                and not any(c.predicate_index is not None or c.subject_indices
                            for c in structure.clauses)):
            # an interrogative with nothing of its own to ask («какое», «а потом что»); one that
            # names its own subject («а для чего он нужен?», «а мыло для чего?») asks on (4.42)
            return True
        asked = await self._database.fetchone(
            "SELECT json_extract(payload_json,'$.content') AS content FROM events WHERE event_id=?",
            (prior["prompt_event_id"],))
        earlier = set(content_lemmas(re.sub(r"<@!?\d+>", " ", str(asked["content"] or "")))) \
            if asked else set()
        return bool(earlier) and earlier == set(content_lemmas(content))

    async def learned_goals(self, event: Event) -> tuple[MeaningGoal, ...]:
        """A valued actual activity can become a goal; cold starts have none."""
        rows = await self._database.fetchall(
            """SELECT p.*, i.action_json FROM agent_preferences p
               JOIN language_meaning_intentions i ON i.intention_id = p.intention_id
               WHERE p.guild_id = ? AND p.channel_id = ? AND p.user_id = ?
                 AND p.updated_at < ? AND p.estimate > 0.10
               ORDER BY p.updated_at DESC LIMIT 128""",
            (event.guild_id, event.channel_id, event.actor_id, event.utc_timestamp.isoformat()),
        )
        return tuple(
            MeaningGoal(
                thought_from_data(json.loads(row["action_json"])),
                "learned_preference:" + row["intention_id"],
                abs(float(row["estimate"])),
            ) for row in rows if row["action_json"]
        )
