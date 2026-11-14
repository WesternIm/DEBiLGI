"""Learn choices between meanings from consequences of actually sent actions.

No response strings, desired topics, or intrinsic positive action values live
here. Cold starts explore equally; nonzero values come only from attributed
human feedback. The preference is selected and stored before surface wording.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
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


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


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
    return hashlib.sha256(_json((proposal.operation, atoms)).encode()).hexdigest()[:24]


class MeaningIntentRepository:
    def __init__(
        self,
        database: Database,
        *,
        exploration_rate: float = 0.10,
        feedback_window: timedelta = timedelta(minutes=10),
    ) -> None:
        if not 0 <= exploration_rate <= 1:
            raise ValueError("exploration_rate must be between zero and one")
        self._database = database
        self._exploration_rate = exploration_rate
        self._feedback_window = feedback_window

    async def choose(
        self,
        event: Event,
        proposals: tuple[DerivedPlan, ...],
        objective: DialogueObjective,
        *,
        parent_intent_id: str,
        state_features: tuple[tuple[str, float], ...] = (),
    ) -> MeaningSelection | None:
        if not proposals or event.channel_id is None or event.actor_id is None:
            return None
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
            """SELECT i.behavior_key, i.operation, c.outcome_type
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

        def relevance(row) -> float:
            previous = json.loads(row["state_json"])
            common = state.keys() & previous.keys()
            distance = sum(abs(state[key] - previous[key]) for key in common)
            return 1.0 / (1.0 + distance / len(common)) if common else 1.0

        for proposal in proposals:
            key = behavior_key(proposal)
            exact = [row for row in rows if row["behavior_key"] == key]
            samples = exact or [row for row in rows if row["operation"] == proposal.operation]
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
            effects = [row for row in consequences if row["behavior_key"] == key] or [
                row for row in consequences if row["operation"] == proposal.operation
            ]
            frequencies: dict[str, int] = {}
            for row in effects:
                kind = row["outcome_type"]
                frequencies[kind] = frequencies.get(kind, 0) + 1
            alternatives.append(
                {
                    "behavior_key": key,
                    "plan_key": proposal.plan.key,
                    "operation": proposal.operation,
                    "expected_reward": value,
                    "expected_outcome": preferred,
                    "sample_count": len(samples),
                    "value_origin": "exact_experience"
                    if exact
                    else "operation_experience"
                    if samples
                    else "untrained",
                    "tie": tie,
                    "experience_ids": tuple(row["intention_id"] for row in samples),
                    "predicted_consequences": {
                        kind: count / len(effects) for kind, count in frequencies.items()
                    },
                }
            )
        draw = int(hashlib.sha256((event.event_id + ":explore").encode()).hexdigest()[:8], 16)
        exploring = not rows or draw / 2**32 < self._exploration_rate
        selected = max(
            range(len(alternatives)),
            key=lambda i: (
                0.0 if exploring else alternatives[i]["expected_reward"],
                alternatives[i]["tie"],
            ),
        )
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
                _json(asdict(proposal.plan)),
                option["expected_reward"],
                option["expected_outcome"],
                option["sample_count"],
                _json({"exploring": exploring, "alternatives": alternatives}),
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
        )

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
        return cursor.rowcount == 1

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
            """SELECT intention_id, plan_json, response_text FROM language_meaning_intentions
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
        claim = json.loads(prior["plan_json"])["claim"]
        outcome = feedback.kind
        effect = None
        reward = feedback.reward if feedback.is_learnable() else None
        if event.event_type in {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED}:
            content = str(event.payload.get("content", ""))
            words = set(tokenize(content.casefold()))
            # "Так да или нет?" is a demand for an answer, not approval merely
            # because the affirmative token occurs in it.
            if "?" in content or words & {"или", "либо"}:
                reward = None
                outcome = "follow_up_question"
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
                    effect = _json(asdict(exchange.bound_proposition))
                    reward = None
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

    async def learned_goals(self, event: Event) -> tuple[MeaningGoal, ...]:
        """A valued actual activity can become a goal; cold starts have none."""
        rows = await self._database.fetchall(
            """SELECT intention_id, action_json, reward FROM language_meaning_intentions
               WHERE guild_id = ? AND channel_id = ? AND user_id = ?
                 AND status = 'observed' AND reward != 0 AND action_json IS NOT NULL
                 AND observed_at < ? AND sent_at < ?
               ORDER BY observed_at DESC LIMIT 128""",
            (
                event.guild_id,
                event.channel_id,
                event.actor_id,
                event.utc_timestamp.isoformat(),
                event.utc_timestamp.isoformat(),
            ),
        )
        if not rows:
            return ()
        value = sum(float(row["reward"]) for row in rows) / len(rows)
        # A rejected answer cannot justify wishing away the whole activity.
        # Only an activity with a positive observed value supplies a goal.
        if value <= 0.10:
            return ()
        action = thought_from_data(json.loads(rows[0]["action_json"]))
        return (
            MeaningGoal(
                action,
                "learned_preference:" + rows[0]["intention_id"],
                abs(value),
            ),
        )
