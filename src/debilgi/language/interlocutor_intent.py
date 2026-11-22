"""Infer the partner's desired update from exchanges, independently of rewards.

Clusters are identities of observed request/continuation graphs, not an authored
catalogue of intentions. Human demonstrations are weak evidence; only feedback
to a confirmed sent meaning intention is relationship-specific experience.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import timedelta
from functools import lru_cache

from debilgi.events.models import Event
from debilgi.experience.feedback import DEFAULT_FEEDBACK_ATTRIBUTOR
from debilgi.language.communication import MessagePlan, reader_goal
from debilgi.language.content import content_lemmas
from debilgi.language.generative_planner import DerivedPlan, SemanticEvidence
from debilgi.language.morphology import DEFAULT_MORPHOLOGY
from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    SemanticDependency,
)
from debilgi.language.structured_utterance import analyze_structured_utterance
from debilgi.language.surface_coherence import evaluate_surface
from debilgi.language.word_relations import WordRelation
from debilgi.storage.database import Database

MAX_INTENT_CHARACTERS = 2048


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _key(value) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()[:24]


@lru_cache(maxsize=8192)
def intent_features(text: str) -> frozenset[str]:
    from debilgi.language.dialogue_learning import exchange_analysis

    structure, act, _ = exchange_analysis(text[:MAX_INTENT_CHARACTERS])
    features = {f"act:{act}"}
    for clause in structure.clauses:
        features.add(f"clause:{clause.clause_kind}")
        features.add(f"polarity:{clause.negated}")
        if clause.implicit_subject:
            features.add(f"participant:{clause.implicit_subject}")
        if clause.predicate_index is not None:
            token = structure.tokens[clause.predicate_index]
            features.add(f"predicate:{token.lemma}")
        for index in clause.subject_indices:
            token = structure.tokens[index]
            person = dict(token.grammatical_features).get("person")
            features.add(f"participant:{person or token.part_of_speech}")
        for modal in clause.modal_lemmas:
            features.add(f"modal:{modal}")
    for token in structure.tokens:
        if token.semantic_role == "question_target":
            features.add(f"question:{token.lemma}")
        if token.part_of_speech in {"noun", "adjective", "adverb"}:
            features.add(f"topic:{token.lemma}")
        if token.part_of_speech:
            features.add(f"role:{token.semantic_role}:{token.part_of_speech}")
    return frozenset(features)


@lru_cache(maxsize=8192)
def response_signature(prompt: str, answer: str) -> dict:
    """A goal hypothesis describes what a reply changed, not its surface text."""
    from debilgi.language.dialogue_learning import exchange_analysis

    source, _, _ = exchange_analysis(prompt[:MAX_INTENT_CHARACTERS])
    target, act, _ = exchange_analysis(answer[:MAX_INTENT_CHARACTERS])
    source_predicates = tuple(
        source.tokens[c.predicate_index].lemma
        for c in source.clauses if c.predicate_index is not None
    )
    goal = reader_goal(source)
    question = next((t for t in source.tokens if t.semantic_role == "question_target"), None)
    question_preposition = (
        source.tokens[question.index - 1].lemma if question is not None and question.index
        and source.tokens[question.index - 1].part_of_speech == "preposition" else None
    )
    first_person = any(
        c.implicit_subject == "speaker"
        or any(target.tokens[i].lemma in {"я", "мы"} for i in c.subject_indices)
        for c in target.clauses
    )
    # A nominal state question can omit an explicit addressee in Russian.
    nominal_state = bool(
        not source_predicates and goal.question_word and goal.topics
        and any(t.part_of_speech in {"adverb", "adjective"} for t in target.tokens)
        and set(goal.topics) & {t.lemma for t in target.tokens}
    )
    return {
        "request_predicates": source_predicates,
        "requested_slot": goal.question_word,
        "requested_case": dict(question.grammatical_features).get("case") if question else None,
        "question_preposition": question_preposition,
        "response_act": act,
        "response_predicates": tuple(
            "$request" if target.tokens[c.predicate_index].lemma in source_predicates
            else target.tokens[c.predicate_index].lemma
            for c in target.clauses if c.predicate_index is not None
        ),
        "response_roles": tuple(sorted({
            (t.semantic_role, t.part_of_speech) for t in target.tokens
            if t.part_of_speech and t.semantic_role != "pragmatic_marker"
        })),
        "self_directed": bool(goal.addressed and first_person or nominal_state),
        "response_asserted": bool(target.clauses) and all(
            c.clause_kind == "statement" and not c.negated and not c.modal_lemmas
            and not c.scopes and c.semantic_time.tense != "future" for c in target.clauses
        ),
    }


@dataclass(frozen=True, slots=True)
class InterlocutorHypothesis:
    cluster_id: str
    signature: dict
    confidence: float
    similarity: float
    observed_count: int
    live_count: int
    evidence_ids: tuple[str, ...]
    outcomes: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class ClarificationPlan:
    proposal: DerivedPlan
    slots: tuple[dict, ...]
    focus_surface: str
    focus_lemma: str
    focus_features: tuple[tuple[str, str], ...]
    mode: str
    construction_key: str
    evidence_ids: tuple[str, ...]
    reason: str


def lexical_slot_supported(
    governor: str, dependent: str, role: str, relations: tuple[WordRelation, ...],
    *, rejected: frozenset[tuple[str, str, str]] = frozenset(),
) -> bool:
    """Exact exceptions precede distributional classes, which need two anchors.

    Sharing a POS/case alone cannot make a table an object of affection. An
    unseen edge needs a predicate neighbour sharing two lexical dependents.
    Negative corrected edges veto both exact and generalized positive evidence.
    """
    if (governor, dependent, role) in rejected:
        return False
    edges: dict[str, set[str]] = defaultdict(set)
    for relation in relations:
        if relation.relation == role and relation.count > 0:
            edges[relation.governor].add(relation.dependent)
    known = edges[governor]
    if dependent in known:
        return True
    return any(
        dependent in partners and len(known & partners) >= 2
        for neighbour, partners in edges.items() if neighbour != governor
    )


def _similarity(left: frozenset[str], right: frozenset[str]) -> float:
    def weight(feature):
        return 5 if feature.startswith("predicate:") else 3 if feature.startswith(
            ("question:", "participant:", "topic:")
        ) else 1

    return sum(weight(f) for f in left & right) / max(
        1, sum(weight(f) for f in left | right)
    )


def goal_projection(signature: dict) -> dict:
    """Cluster by the requested update, without a corpus answer's vocabulary.

    Reading and drawing can both fill a demonstrated activity slot. Their
    predicates remain evidence attached to the cluster, not different goals.
    """
    return {
        key: signature.get(key) for key in (
            "request_predicates", "requested_slot", "requested_case",
            "question_preposition", "response_act", "self_directed",
        )
    } | {
        "retains_request_predicate": "$request" in signature["response_predicates"],
        "supplies_predicate": bool(set(signature["response_predicates"]) - {"$request"}),
    }


class InterlocutorIntentRepository:
    def __init__(self, database: Database):
        self._database = database

    async def observe_pair(self, event: Event, parent, current) -> None:
        features = intent_features(parent["content"])
        signature = response_signature(parent["content"], current["content"])
        async with self._database.transaction() as connection:
            await connection.execute(
                """INSERT OR IGNORE INTO language_interlocutor_examples
                   (example_id, guild_id, channel_id, user_id, prompt_event_id,
                    response_event_id, source_kind, features_json, signature_json, observed_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'observed', ?, ?, ?)""",
                (event.event_id, event.guild_id, event.channel_id, parent["actor_id"],
                 parent["source_id"], event.event_id, _json(sorted(features)),
                 _json(signature), event.utc_timestamp.isoformat()),
            )
        # A same-speaker tail such as a reciprocal pronoun can refer to the
        # preceding report. Retain at most two raw turns, never inferred text.
        prior_turns = await self._database.fetchall(
            """SELECT content FROM language_discourse_turns
               WHERE guild_id = ? AND channel_id = ? AND actor_id = ?
                 AND observed_at < ? AND observed_at >= ?
               ORDER BY observed_at DESC LIMIT 2""",
            (event.guild_id, event.channel_id, parent["actor_id"], parent["observed_at"],
             (event.utc_timestamp - timedelta(minutes=5)).isoformat()),
        )
        question_context = " ".join([parent["content"], *(r["content"] for r in prior_turns)])
        await self._learn_questions(event, question_context, current["content"])

    async def observe_continuation(self, event: Event, parent, current) -> None:
        rows = await self._database.fetchall(
            """SELECT e.*, p.content AS prompt, a.content AS answer
               FROM language_interlocutor_examples e
               JOIN language_discourse_turns p ON p.source_id = e.prompt_event_id
               JOIN language_discourse_turns a ON a.source_id = e.response_event_id
               WHERE e.response_event_id = ? AND e.user_id = ?
                 AND e.continuation_event_id IS NULL AND e.source_kind = 'observed'""",
            (parent["source_id"], event.actor_id),
        )
        for row in rows:
            feedback = DEFAULT_FEEDBACK_ATTRIBUTOR.attribute(event)
            continuation = set(content_lemmas(current["content"]))
            related = continuation & set(
                content_lemmas(row["answer"]) + content_lemmas(row["prompt"])
            )
            outcome = (
                "rejected" if feedback.is_learnable() and feedback.reward < 0
                else "confirmed" if feedback.is_learnable() and feedback.reward > 0
                else "topic_continued" if related else "topic_changed"
            )
            await self._database.execute(
                """UPDATE language_interlocutor_examples
                   SET outcome = ?, continuation_event_id = ? WHERE example_id = ?
                     AND continuation_event_id IS NULL""",
                (outcome, event.event_id, row["example_id"]),
            )

    async def observe_live(self, event: Event, prior, outcome: str, reward: float | None) -> None:
        """Called only after MeaningIntentRepository found a real sent target."""
        prompt = await self._database.fetchone(
            "SELECT payload_json FROM events WHERE event_id = ?", (prior["prompt_event_id"],)
        )
        if prompt is None or not prior["response_text"]:
            return
        content = str(json.loads(prompt["payload_json"]).get("content", ""))
        if reward is None and set(content_lemmas(str(event.payload.get("content", "")))) & set(
            content_lemmas(content) + content_lemmas(prior["response_text"])
        ):
            outcome = "topic_continued"
        await self._database.execute(
            """INSERT OR IGNORE INTO language_interlocutor_examples
               (example_id, guild_id, channel_id, user_id, prompt_event_id,
                intention_id, source_kind, features_json, signature_json,
                outcome, continuation_event_id, observed_at)
               VALUES (?, ?, ?, ?, ?, ?, 'live', ?, ?, ?, ?, ?)""",
            ("live:" + event.event_id, event.guild_id, event.channel_id, event.actor_id,
             prior["prompt_event_id"], prior["intention_id"],
             _json(sorted(intent_features(content))),
             _json(response_signature(content, prior["response_text"])),
             "rejected" if reward is not None and reward < 0
             else "confirmed" if reward is not None and reward > 0 else outcome,
             event.event_id, event.utc_timestamp.isoformat()),
        )

    async def infer(self, event: Event, prompt: str) -> tuple[InterlocutorHypothesis, ...]:
        features = intent_features(prompt)
        rows = await self._database.fetchall(
            """SELECT * FROM language_interlocutor_examples
               WHERE guild_id = ? AND observed_at < ?
                 AND (source_kind = 'observed' OR (channel_id = ? AND user_id = ?))
                 ORDER BY observed_at DESC, example_id LIMIT 2048""",
            (event.guild_id, event.utc_timestamp.isoformat(), event.channel_id, event.actor_id),
        )
        neighbours = []
        for row in rows:
            similarity = _similarity(features, frozenset(json.loads(row["features_json"])))
            if similarity >= 0.30:
                neighbours.append((similarity, row))
        # A concrete correction to the same request outranks broad regularities.
        exact_live = [item for item in neighbours if item[0] >= 0.98
                      and item[1]["source_kind"] == "live"]
        selected = exact_live or sorted(neighbours, key=lambda item: -item[0])[:32]
        groups = defaultdict(list)
        for similarity, row in selected:
            signature = json.loads(row["signature_json"])
            # Concrete relationship experience remains an exception at its
            # full response graph; raw demonstrations share an abstract goal.
            key = row["signature_json"] if exact_live else _json(goal_projection(signature))
            groups[key].append((similarity, row))
        hypotheses = []
        for raw, samples in groups.items():
            signatures = [json.loads(row["signature_json"]) for _, row in samples]
            signature = dict(signatures[0])
            signature["response_predicates"] = sorted({
                p for sample in signatures for p in sample["response_predicates"]
            })
            signature["response_roles"] = sorted({
                tuple(role) for sample in signatures for role in sample["response_roles"]
            })
            live = sum(row["source_kind"] == "live" for _, row in samples)
            outcomes = defaultdict(int)
            for _, row in samples:
                outcomes[row["outcome"]] += 1
            similarity = max(sim for sim, _ in samples)
            support = sum(sim * (3 if row["source_kind"] == "live" else 1)
                          for sim, row in samples)
            rejection = outcomes["rejected"] / len(samples)
            continuity = (outcomes["topic_continued"] + outcomes["confirmed"]) / len(samples)
            confidence = min(0.92 if live else 0.64,
                             similarity * (0.40 + 0.20 * support / (support + 2)
                                           + 0.15 * continuity)) * (1 - rejection)
            hypotheses.append(InterlocutorHypothesis(
                _key(json.loads(raw)), signature, confidence, similarity,
                len(samples) - live, live,
                tuple(row["example_id"] for _, row in samples[:8]),
                tuple(sorted(outcomes.items())),
            ))
        return tuple(sorted(hypotheses, key=lambda h: (-h.confidence, -h.similarity, h.cluster_id)))

    async def clarification_allowed(self, event: Event) -> bool:
        """One consecutive repair per relationship, based on confirmed sends.

        Prepared/withheld proposals do not consume this budget. Even a changed
        wording or a reply to the repair cannot recursively restart it.
        """
        if event.payload.get("author_is_bot") or event.actor_id is None:
            return False
        row = await self._database.fetchone(
            """SELECT operation, plan_json FROM language_meaning_intentions
               WHERE guild_id = ? AND channel_id = ? AND user_id = ?
                 AND status IN ('sent', 'observed') AND sent_at < ?
                 AND (sent_at >= ? OR response_message_id = ?)
               ORDER BY (response_message_id = ?) DESC, sent_at DESC, intention_id LIMIT 1""",
            (event.guild_id, event.channel_id, event.actor_id,
             event.utc_timestamp.isoformat(),
             (event.utc_timestamp - timedelta(minutes=10)).isoformat(),
             event.payload.get("reply_to_message_id"), event.payload.get("reply_to_message_id")),
        )
        return row is None or (
            not row["operation"].startswith("clarify_interlocutor:")
            and json.loads(row["plan_json"])["claim"]["clause_kind"] != "question"
        )

    async def grounded_agent_plans(self, event: Event, prompt: str, hypotheses, goals=()):
        """Compatibility adapter: operational facts now belong to PropositionMemory.

        Goals and corpus request clusters cannot prove current execution. The
        evidence planner retrieves canonical self claims with their valid time.
        """
        from debilgi.language.proposition_memory import PropositionMemory

        memory = PropositionMemory(self._database)
        result = await memory.retrieve_claims(event, prompt, hypotheses=hypotheses)
        from debilgi.language.canonical_claims import self_query
        from debilgi.language.generative_planner import plans_for_claim_result

        return (plans_for_claim_result(result, analyze_structured_utterance(prompt))
                if self_query(result.query) else ())

    async def predicate_children(self, query) -> tuple[dict[str, tuple[str, ...]], frozenset[str]]:
        """One raw question/answer edge, never an edge inferred from another edge.

        Corpus examples teach the query relation, not a fact about either
        participant. Explicitly rejected edges veto morphological defaults.
        """
        rows = await self._database.fetchall(
            """SELECT e.example_id, e.signature_json, e.outcome, p.content AS prompt,
                      a.content AS answer FROM language_interlocutor_examples e
               LEFT JOIN language_discourse_turns p ON p.source_id = e.prompt_event_id
               LEFT JOIN language_discourse_turns a ON a.source_id = e.response_event_id
               WHERE e.guild_id = ? AND e.observed_at < ? AND e.source_kind = 'observed'
               ORDER BY e.observed_at DESC, e.example_id LIMIT 2048""",
            (query.guild_id, query.snapshot_at),
        )
        children = defaultdict(list)
        rejected = set()
        for row in rows:
            signature = json.loads(row["signature_json"])
            if "response_asserted" not in signature and row["prompt"] and row["answer"]:
                signature = response_signature(row["prompt"], row["answer"])
            if (signature.get("requested_slot") != "что"
                    or query.predicate_parent not in signature.get("request_predicates", ())
                    or signature.get("response_act") in {"question", "clarification"}):
                continue
            for predicate in signature.get("response_predicates", ()):
                if predicate in {"$request", "делать", "заниматься"}:
                    continue
                if row["outcome"] == "rejected":
                    rejected.add(predicate)
                elif signature.get("response_asserted"):
                    children[predicate].append(row["example_id"])
        return {p: tuple(ids[:8]) for p, ids in children.items()}, frozenset(rejected)

    async def rejected_relations(self, event: Event) -> frozenset[tuple[str, str, str]]:
        from debilgi.language.word_relations import observe_word_relations

        rows = await self._database.fetchall(
            """SELECT response_text FROM language_meaning_intentions
               WHERE guild_id = ? AND channel_id = ? AND user_id = ?
                 AND status = 'observed' AND reward < 0 AND observed_at < ?
               ORDER BY observed_at DESC LIMIT 128""",
            (event.guild_id, event.channel_id, event.actor_id, event.utc_timestamp.isoformat()),
        )
        return frozenset((r.governor, r.dependent, r.relation)
                         for row in rows for r in observe_word_relations(row["response_text"] or "")
                         if r.relation in {"obj", "obl", "iobj"})

    async def _learn_questions(self, event: Event, prompt: str, answer: str) -> None:
        """Extract productive nominal questions and detached echo operators."""
        source = analyze_structured_utterance(prompt[:MAX_INTENT_CHARACTERS])
        structure = analyze_structured_utterance(answer[:MAX_INTENT_CHARACTERS])
        prompt_terms = {t.lemma for t in source.tokens if t.part_of_speech in {"noun", "verb"}}
        constructions = []
        for clause in structure.clauses:
            if clause.clause_kind != "question":
                continue
            tokens = [t for t in structure.tokens[clause.token_start:clause.token_end]
                      if t.surface[:1].isalnum()]
            nouns = [t for t in tokens if t.part_of_speech == "noun"]
            question = (structure.tokens[clause.question_target_index]
                        if clause.question_target_index is not None else None)
            if (clause.predicate_index is None and clause.question_target_index is not None
                    and (question.part_of_speech == "adjective" or question.lemma == "что")
                    and len(nouns) == 1 and len(tokens) <= 4
                    and nouns[0].lemma in prompt_terms
                    and all(t == nouns[0] or t.index == clause.question_target_index
                            or t.part_of_speech in {"preposition", "particle", "conjunction"}
                            for t in tokens)):
                constructions.append(("nominal", [
                    {"binding": "focus", "features": dict(t.grammatical_features)}
                    if t == nouns[0] else {
                        "binding": (
                            "question" if t.index == clause.question_target_index else "grammar"
                        ),
                        "lemma": t.lemma, "surface": t.surface,
                        "features": dict(t.grammatical_features),
                    } for t in tokens
                ]))
        # Detached prepositional operators in echo/alternative questions are
        # learned from their use. No authored metalinguistic word list exists.
        comma = next((t.index for t in structure.tokens if t.surface == ","), None)
        if comma is not None:
            prefix = structure.tokens[:comma]
            tail = structure.tokens[comma + 1:]
            if (len(prefix) == 2 and prefix[0].part_of_speech == "preposition"
                    and prefix[1].part_of_speech == "noun"
                    and prefix[1].lemma not in prompt_terms
                    and any(c.clause_kind == "question" for c in structure.clauses)
                    and len([t for t in tail if t.part_of_speech == "verb"]) >= 2
                    and prompt_terms & {t.lemma for t in tail}
                    and not evaluate_surface(
                        " ".join(t.surface for t in prefix) + "?"
                    ).hard_issues):
                constructions.append(("mention", [
                    {"binding": "grammar", "lemma": t.lemma, "surface": t.surface,
                     "features": dict(t.grammatical_features)} for t in prefix
                ] + [{"binding": "focus", "features": {}}]))
        for mode, slots in constructions:
            await self._database.execute(
                """INSERT OR IGNORE INTO language_clarification_constructions
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (event.event_id, _key((mode, slots)), event.guild_id,
                 _json(slots), mode, event.utc_timestamp.isoformat()),
            )

    async def clarification_plans(
        self, event: Event, prompt: str, *, reason: str,
    ) -> tuple[ClarificationPlan, ...]:
        rows = await self._database.fetchall(
            """SELECT construction_key, slots_json, focus_mode,
                      GROUP_CONCAT(source_event_id) AS evidence_ids, COUNT(*) AS support
               FROM language_clarification_constructions
               WHERE guild_id = ? AND observed_at < ? GROUP BY construction_key
               ORDER BY support DESC, construction_key LIMIT 64""",
            (event.guild_id, event.utc_timestamp.isoformat()),
        )
        structure = analyze_structured_utterance(prompt[:MAX_INTENT_CHARACTERS])
        def focus_priority(token):
            if token.semantic_role == "subject":
                return 0
            if token.semantic_role == "direct_object" and any(
                token.index in clause.direct_object_indices
                and clause.predicate_index is not None
                and dict(structure.tokens[clause.predicate_index].grammatical_features).get(
                    "transitivity"
                ) == "transitive" for clause in structure.clauses
            ):
                return 0
            if token.semantic_role == "action_complement":
                return 1
            return 2 if token.semantic_role == "predicate" else 3

        focus = sorted(
            (t for t in structure.tokens if t.part_of_speech in {"noun", "verb"}
             and t.semantic_role not in {"vocative", "pragmatic_marker"}
             and DEFAULT_MORPHOLOGY.best(t.surface) is not None),
            key=lambda t: (focus_priority(t), -t.index),
        )
        plans = []
        for row in rows:
            token = focus[0] if focus else None
            if token is None or (row["focus_mode"] == "nominal" and token.part_of_speech != "noun"):
                continue
            slots = tuple(json.loads(row["slots_json"]))
            ids = tuple(row["evidence_ids"].split(",")[:8])
            # The semantic object is a *mention* of the user's token. It is
            # neither an assertion about the bot nor a new corpus topic.
            head = ConceptLexeme(token.lemma, token.surface, token.part_of_speech,
                                 token.grammatical_features)
            operator = next((s for s in slots if s.get("lemma")
                             and DEFAULT_MORPHOLOGY.best(s["surface"]).part_of_speech
                             in {"noun", "adjective", "pronoun"}), None)
            if operator is None:
                operator = next((s for s in slots if s.get("lemma")), None)
            if operator is None:
                continue
            thought = ClauseThought(
                ConceptLexeme(operator["lemma"], operator["surface"], "noun"),
                (SemanticDependency("subject", ConceptArgument(head)),),
                clause_kind="question", confidence=0.70,
                evidence=(event.event_id, *ids),
            )
            proposal = DerivedPlan(
                MessagePlan(reader_goal(structure), thought),
                "clarify_interlocutor:" + row["construction_key"],
                (SemanticEvidence(event.event_id, "unresolved_user_intention", prompt, thought),),
            )
            plans.append(ClarificationPlan(
                proposal, slots, token.surface, token.lemma, token.grammatical_features,
                row["focus_mode"], row["construction_key"], ids, reason,
            ))
        return tuple(plans)


def realize_clarification(plan: ClarificationPlan) -> str | None:
    """Fill learned question slots; inflection and agreement belong to pymorphy3."""
    features = dict(plan.focus_features)
    slot = next(s for s in plan.slots if s["binding"] == "focus")
    case = slot["features"].get("case", "nominative")
    # Russian interrogative determiner: here the particle isn't the spatial
    # preposition governing accusative. An ambiguous corpus noun (e.g. neuter)
    # must not propagate an accidental accusative parse to animate plurals.
    operators = tuple(s.get("lemma") for s in plan.slots if s["binding"] != "focus")
    if any(operators[i:i + 2] == ("что", "за") for i in range(len(operators) - 1)):
        case = "nominative"
    number = features.get("number", "singular")
    result = []
    for item in plan.slots:
        if item["binding"] == "focus":
            surface = (
                "«" + plan.focus_surface + "»" if plan.mode == "mention"
                else DEFAULT_MORPHOLOGY.inflect(plan.focus_lemma, case=case, number=number)
            )
        elif item["binding"] == "question" and item["lemma"] == "какой":
            surface = DEFAULT_MORPHOLOGY.inflect(
                item["lemma"], **{
                    name: value for name, value in {
                        "case": case, "number": number,
                        "gender": features.get("gender") if number == "singular" else None,
                    }.items() if value is not None
                },
            )
        else:
            surface = item["surface"]
        if not surface:
            return None
        result.append(surface)
    text = " ".join(result)
    text = text[:1].upper() + text[1:] + "?"
    if evaluate_surface(text).hard_issues:
        return None
    return text


def hypothesis_data(hypotheses: tuple[InterlocutorHypothesis, ...]) -> tuple[dict, ...]:
    return tuple(asdict(hypothesis) for hypothesis in hypotheses[:8])
